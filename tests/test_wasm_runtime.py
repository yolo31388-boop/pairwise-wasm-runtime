"""
游戏内WebAssembly运行时与沙箱系统 - 验收测试
运行方式：python -m pytest tests/test_wasm_runtime.py -q
共 15 个测试用例
"""
import os

import pytest

from runtime.wasm_runtime import (
    WASMRuntime,
    WasmValidationError,
    SandboxViolationError,
    MemoryAccessError,
    FunctionCallError,
    ModuleNotFoundError,
)

WASM_HEADER = b"\x00asm\x01\x00\x00\x00"


def wasm_module(marker: bytes = b"") -> bytes:
    return WASM_HEADER + marker


class TestWASMRuntime:
    def setup_method(self):
        self.system = WASMRuntime()

    # ---------- 模块实例化：验证 + 隔离 ----------

    def test_case_01(self):
        instance = self.system.instantiate_module("game_mod", wasm_module(b"v1"))
        assert instance.name == "game_mod"
        assert self.system.get_module("game_mod").digest == instance.digest

    def test_case_02(self):
        with pytest.raises(WasmValidationError):
            self.system.instantiate_module("bad1", b"#!/bin/sh\nrm -rf /")
        with pytest.raises(WasmValidationError):
            self.system.instantiate_module("bad2", b"\x00asm")

    def test_case_03(self):
        mod_a = self.system.instantiate_module("mod_a", wasm_module(b"a"))
        mod_b = self.system.instantiate_module("mod_b", wasm_module(b"b"))
        assert mod_a.memory_id != mod_b.memory_id
        memory = self.system.memory_manager()
        memory.write(mod_a.memory_id, 0, b"SECRET")
        assert memory.read(mod_b.memory_id, 0, 6) == b"\x00" * 6
        with pytest.raises(MemoryAccessError):
            memory.read("mem:nonexistent", 0, 1)

    # ---------- 沙箱：权限控制 + 系统调用拦截 ----------

    def test_case_04(self, tmp_path):
        inside = tmp_path / "saves" / "slot1.dat"
        inside.parent.mkdir()
        inside.write_bytes(b"data")
        runtime = WASMRuntime({"allowed_paths": {str(tmp_path): "rw"}})
        sandbox = runtime.sandbox()
        assert sandbox.check_file_access(str(inside), "r") is True
        with pytest.raises(SandboxViolationError):
            sandbox.check_file_access("/etc/passwd", "r")
        with pytest.raises(SandboxViolationError):
            sandbox.check_file_access(os.path.join(str(tmp_path), "..", "secret"), "r")

    def test_case_05(self, tmp_path):
        save_file = tmp_path / "save.dat"
        save_file.write_bytes(b"x")
        runtime = WASMRuntime({"allowed_paths": {str(tmp_path): "r"}})
        sandbox = runtime.sandbox()
        assert sandbox.check_file_access(str(save_file), "r") is True
        with pytest.raises(SandboxViolationError):
            sandbox.check_file_access(str(save_file), "w")

    def test_case_06(self):
        runtime = WASMRuntime({"allowed_syscalls": ("clock_gettime",)})
        sandbox = runtime.sandbox()
        assert sandbox.intercept_syscall("clock_gettime") is True
        with pytest.raises(SandboxViolationError):
            sandbox.intercept_syscall("unlink")
        with pytest.raises(SandboxViolationError):
            sandbox.intercept_syscall("execve")

    # ---------- 内存管理：边界检查 + 垃圾回收 ----------

    def test_case_07(self):
        mod = self.system.instantiate_module("mem_mod", wasm_module(b"m"))
        memory = self.system.memory_manager()
        memory.write(mod.memory_id, 16, b"hello")
        assert memory.read(mod.memory_id, 16, 5) == b"hello"
        assert memory.read(mod.memory_id, 15, 1) == b"\x00"

    def test_case_08(self):
        mod = self.system.instantiate_module("bounds_mod", wasm_module(b"b"))
        memory = self.system.memory_manager()
        with pytest.raises(MemoryAccessError):
            memory.read(mod.memory_id, -1, 4)
        with pytest.raises(MemoryAccessError):
            memory.read(mod.memory_id, 65536 - 2, 4)
        with pytest.raises(MemoryAccessError):
            memory.write(mod.memory_id, 65536 - 2, b"abc")

    def test_case_09(self):
        mod_a = self.system.instantiate_module("gc_a", wasm_module(b"a"))
        mod_b = self.system.instantiate_module("gc_b", wasm_module(b"b"))
        memory = self.system.memory_manager()
        assert memory.region_count() == 2
        self.system.unload_module("gc_a")
        self.system.update(0.016)
        assert memory.region_count() == 1
        with pytest.raises(MemoryAccessError):
            memory.read(mod_a.memory_id, 0, 1)
        assert memory.read(mod_b.memory_id, 0, 1) == b"\x00"

    # ---------- 函数调用：参数验证 + 返回值检查 ----------

    def test_case_10(self):
        self.system.instantiate_module("plug", wasm_module(b"p"))
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "system", "rm -rf /")
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "os_remove", "save.dat")
        with pytest.raises(ModuleNotFoundError):
            self.system.function_call("not_loaded", "anything")

    def test_case_11(self):
        self.system.register_host_function(
            "add", lambda a, b: a + b, (int, int), int)
        self.system.instantiate_module("plug", wasm_module(b"p"))
        assert self.system.function_call("plug", "add", 2, 3) == 5
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "add", 2)
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "add", 2, 3, 4)
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "add", "2", 3)

    def test_case_12(self):
        self.system.register_host_function(
            "broken_return", lambda: "not an int", (), int)
        self.system.register_host_function(
            "answer", lambda: 42, (), int)
        self.system.instantiate_module("plug", wasm_module(b"p"))
        assert self.system.function_call("plug", "answer") == 42
        with pytest.raises(FunctionCallError):
            self.system.function_call("plug", "broken_return")

    # ---------- 模块缓存：命中 + 失效 + 容量限制 ----------

    def test_case_13(self):
        bytecode = wasm_module(b"cached")
        first = self.system.instantiate_module("cached_mod", bytecode)
        second = self.system.instantiate_module("cached_mod", bytecode)
        assert first is second
        assert len(self.system.module_cache()) == 1

    def test_case_14(self):
        v1 = wasm_module(b"version-1")
        v2 = wasm_module(b"version-2")
        old = self.system.instantiate_module("updatable", v1)
        new = self.system.instantiate_module("updatable", v2)
        assert old.digest != new.digest
        assert self.system.module_cache().get("updatable", new.digest) is new
        assert self.system.module_cache().get("updatable", old.digest) is None

    def test_case_15(self):
        runtime = WASMRuntime({"cache_max_entries": 2})
        m1 = runtime.instantiate_module("m1", wasm_module(b"1"))
        runtime.instantiate_module("m2", wasm_module(b"2"))
        runtime.instantiate_module("m3", wasm_module(b"3"))
        cache = runtime.module_cache()
        assert len(cache) == 2
        assert cache.get("m1", m1.digest) is None
        assert cache.get("m3", runtime.get_module("m3").digest) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
