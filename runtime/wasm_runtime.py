"""
游戏内WebAssembly运行时与沙箱系统 - 核心模块

提供：
- instantiate_module(): 带验证与隔离的模块实例化
- sandbox(): 文件系统权限控制与系统调用拦截
- memory_manager(): 带边界检查与垃圾回收的线性内存管理
- function_call(): 带参数验证与返回值检查的宿主函数调用
- module_cache(): 带失效与容量限制的模块缓存
"""
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

WASM_MAGIC = b"\x00asm"
WASM_VERSION_1 = b"\x01\x00\x00\x00"
PAGE_SIZE = 65536  # WebAssembly 页大小 64KiB


class WasmValidationError(ValueError):
    """无效的 wasm 模块字节码"""


class SandboxViolationError(PermissionError):
    """沙箱权限违规"""


class MemoryAccessError(IndexError):
    """越界内存访问"""


class FunctionCallError(TypeError):
    """非法函数调用"""


class ModuleNotFoundError(KeyError):
    """模块未实例化"""


@dataclass
class Config:
    allowed_paths: Dict[str, str] = None      # 路径 -> "r" / "rw"
    allowed_syscalls: Tuple[str, ...] = ()
    cache_max_entries: int = 8
    max_memory_pages: int = 16


@dataclass
class ModuleInstance:
    name: str
    digest: str
    size: int
    memory_id: str
    instantiated_at: float


class Sandbox:
    """文件系统权限控制 + 系统调用拦截"""

    def __init__(self, allowed_paths: Optional[Dict[str, str]] = None,
                 allowed_syscalls: Optional[List[str]] = None):
        self._permissions = {
            os.path.abspath(path): perm for path, perm in (allowed_paths or {}).items()
        }
        self._allowed_syscalls = set(allowed_syscalls or [])

    def check_file_access(self, path: str, mode: str = "r") -> bool:
        """校验文件访问是否被允许；违规抛出 SandboxViolationError"""
        target = os.path.abspath(path)
        for allowed, perm in self._permissions.items():
            if target == allowed or target.startswith(allowed + os.sep):
                if "w" in mode and "w" not in perm:
                    raise SandboxViolationError(
                        "路径 %r 只读，禁止写入" % path)
                return True
        raise SandboxViolationError("路径 %r 不在沙箱允许列表中" % path)

    def intercept_syscall(self, name: str) -> bool:
        """拦截系统调用；不在白名单中的调用抛出 SandboxViolationError"""
        if name not in self._allowed_syscalls:
            raise SandboxViolationError("系统调用 %r 被沙箱拦截" % name)
        return True


class MemoryManager:
    """按模块隔离的线性内存，带边界检查与垃圾回收"""

    def __init__(self, max_pages_per_module: int = 16):
        self.max_pages_per_module = max_pages_per_module
        self._regions: Dict[str, bytearray] = {}

    def allocate(self, memory_id: str, pages: int = 1) -> None:
        if pages < 1 or pages > self.max_pages_per_module:
            raise MemoryAccessError(
                "内存页数 %d 超出限制 [1, %d]" % (pages, self.max_pages_per_module))
        self._regions[memory_id] = bytearray(PAGE_SIZE * pages)

    def free(self, memory_id: str) -> None:
        self._regions.pop(memory_id, None)

    def _region(self, memory_id: str) -> bytearray:
        if memory_id not in self._regions:
            raise MemoryAccessError("内存区域 %r 不存在或已回收" % memory_id)
        return self._regions[memory_id]

    def read(self, memory_id: str, offset: int, length: int) -> bytes:
        region = self._region(memory_id)
        if offset < 0 or length < 0 or offset + length > len(region):
            raise MemoryAccessError(
                "越界读取: offset=%d length=%d size=%d" % (offset, length, len(region)))
        return bytes(region[offset:offset + length])

    def write(self, memory_id: str, offset: int, data: bytes) -> None:
        region = self._region(memory_id)
        if offset < 0 or offset + len(data) > len(region):
            raise MemoryAccessError(
                "越界写入: offset=%d length=%d size=%d" % (offset, len(data), len(region)))
        region[offset:offset + len(data)] = data

    def gc(self, live_memory_ids: List[str]) -> int:
        """回收不再被任何存活模块引用的内存，返回回收的区域数"""
        live = set(live_memory_ids)
        dead = [mid for mid in self._regions if mid not in live]
        for mid in dead:
            del self._regions[mid]
        return len(dead)

    def region_count(self) -> int:
        return len(self._regions)


class ModuleCache:
    """按内容哈希校验的 LRU 模块缓存，带失效与容量限制"""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max(1, max_entries)
        self._entries: "OrderedDict[str, Tuple[str, ModuleInstance]]" = OrderedDict()

    def get(self, name: str, digest: str) -> Optional[ModuleInstance]:
        """命中且内容未变化时返回缓存实例，否则返回 None"""
        entry = self._entries.get(name)
        if entry is None:
            return None
        cached_digest, instance = entry
        if cached_digest != digest:
            self.invalidate(name)  # 模块已更新，旧缓存失效
            return None
        self._entries.move_to_end(name)
        return instance

    def put(self, name: str, digest: str, instance: ModuleInstance) -> None:
        self._entries[name] = (digest, instance)
        self._entries.move_to_end(name)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # 淘汰最久未使用的条目

    def invalidate(self, name: str) -> None:
        self._entries.pop(name, None)

    def __len__(self) -> int:
        return len(self._entries)


class WASMRuntime:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._sandbox = Sandbox(
            allowed_paths=self.config.get("allowed_paths"),
            allowed_syscalls=self.config.get("allowed_syscalls"),
        )
        self._memory = MemoryManager(
            max_pages_per_module=self.config.get("max_memory_pages", 16))
        self._cache = ModuleCache(
            max_entries=self.config.get("cache_max_entries", 8))
        self._modules: Dict[str, ModuleInstance] = {}
        self._host_functions: Dict[str, Tuple[Callable, Tuple[type, ...], Optional[type]]] = {}
        for name, spec in self.config.get("host_functions", {}).items():
            func, arg_types, return_type = spec
            self.register_host_function(name, func, arg_types, return_type)

    # ---------- 模块实例化（验证 + 隔离） ----------

    @staticmethod
    def validate_module(wasm_bytes: bytes) -> None:
        if not isinstance(wasm_bytes, (bytes, bytearray)):
            raise WasmValidationError("模块必须是字节串")
        if len(wasm_bytes) < 8:
            raise WasmValidationError("模块太短，缺少文件头")
        if bytes(wasm_bytes[:4]) != WASM_MAGIC:
            raise WasmValidationError("无效的 wasm 魔数")
        if bytes(wasm_bytes[4:8]) != WASM_VERSION_1:
            raise WasmValidationError("不支持的 wasm 版本")

    def instantiate_module(self, name: str, wasm_bytes: bytes,
                           memory_pages: int = 1) -> ModuleInstance:
        """验证并实例化模块；每个实例拥有独立隔离的线性内存"""
        self.validate_module(wasm_bytes)
        digest = hashlib.sha256(bytes(wasm_bytes)).hexdigest()
        cached = self._cache.get(name, digest)
        if cached is not None:
            self._modules[name] = cached
            return cached
        memory_id = "mem:%s:%s" % (name, digest[:12])
        self._memory.allocate(memory_id, memory_pages)
        instance = ModuleInstance(
            name=name, digest=digest, size=len(wasm_bytes),
            memory_id=memory_id, instantiated_at=time.time())
        self._modules[name] = instance
        self._cache.put(name, digest, instance)
        self._history.append(("instantiate", name, digest))
        return instance

    def unload_module(self, name: str) -> None:
        instance = self._modules.pop(name, None)
        if instance is None:
            raise ModuleNotFoundError(name)
        self._cache.invalidate(name)
        self._memory.free(instance.memory_id)

    def get_module(self, name: str) -> ModuleInstance:
        if name not in self._modules:
            raise ModuleNotFoundError(name)
        return self._modules[name]

    # ---------- 子系统访问 ----------

    def sandbox(self) -> Sandbox:
        return self._sandbox

    def memory_manager(self) -> MemoryManager:
        return self._memory

    def module_cache(self) -> ModuleCache:
        return self._cache

    # ---------- 函数调用（参数验证 + 返回值检查） ----------

    def register_host_function(self, name: str, func: Callable,
                               arg_types: Tuple[type, ...],
                               return_type: Optional[type]) -> None:
        self._host_functions[name] = (func, tuple(arg_types), return_type)

    def function_call(self, module_name: str, func_name: str, *args: Any) -> Any:
        """仅允许调用已注册的宿主函数，并校验参数与返回值类型"""
        if module_name not in self._modules:
            raise ModuleNotFoundError(module_name)
        if func_name not in self._host_functions:
            raise FunctionCallError(
                "函数 %r 未注册，禁止调用任意系统函数" % func_name)
        func, arg_types, return_type = self._host_functions[func_name]
        if len(args) != len(arg_types):
            raise FunctionCallError(
                "参数数量错误: 期望 %d 个，实际 %d 个" % (len(arg_types), len(args)))
        for index, (arg, expected) in enumerate(zip(args, arg_types)):
            if not isinstance(arg, expected):
                raise FunctionCallError(
                    "第 %d 个参数类型错误: 期望 %s，实际 %s"
                    % (index, expected.__name__, type(arg).__name__))
        result = func(*args)
        if return_type is None:
            if result is not None:
                raise FunctionCallError("返回值类型错误: 期望 None")
        elif not isinstance(result, return_type):
            raise FunctionCallError(
                "返回值类型错误: 期望 %s，实际 %s"
                % (return_type.__name__, type(result).__name__))
        self._history.append(("call", module_name, func_name))
        return result

    # ---------- 生命周期 ----------

    def update(self, dt: float):
        live = [inst.memory_id for inst in self._modules.values()]
        self._memory.gc(live)

    def reset(self):
        self._state = {}
        self._history = []
        for instance in list(self._modules.values()):
            self._memory.free(instance.memory_id)
        self._modules = {}
        self._cache = ModuleCache(
            max_entries=self.config.get("cache_max_entries", 8))
