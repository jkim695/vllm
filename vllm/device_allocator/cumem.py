# SPDX-License-Identifier: Apache-2.0

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import dataclasses
import gc
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from vllm.utils import is_pin_memory_available
from vllm.logger import init_logger

logger = init_logger(__name__)

def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), \
        f"Unexpected filename: {filename} for library {lib_name}"
    return path


cumem_available = False
try:
    from vllm.cumem_allocator import (init_module, python_create_and_map,
                                      python_unmap_and_release)
    from vllm.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary)
    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    # rocm platform does not support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = Tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: Optional[torch.Tensor] = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[int],
                               int], python_free_func: Callable[[int, int],
                                                                None]
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, 'my_malloc', 'my_free')
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
        python_malloc_fn: Callable[[int], int],
        python_free_func: Callable[[int, int], None]) -> None:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc

class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """
    instance: "CuMemAllocator" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """
        CuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, \
            ("Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates.")

        self.pointer_to_data: Dict[int, AllocationData] = {}
        self.sleeping_pointer_to_data: Dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: Dict[str, Any] = {}
        # Track pointers that have been freed during sleep to prevent double-free
        self._freed_pointers: set = set()
        self._is_sleeping: bool = False

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag)
        logger.debug(f"Allocated {allocation_handle[1]} bytes for tag {self.current_tag} at ptr {py_d_mem}")
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        
        # If we're sleeping and this pointer was already freed, return a dummy handle
        if self._is_sleeping and ptr in self._freed_pointers:
            logger.debug(f"Skipping already freed pointer {ptr} during sleep mode")
            # Return a dummy handle to prevent crash
            return (0, 0, ptr, 0)
        
        # Check if pointer exists in our registry
        data = self.pointer_to_data.pop(ptr, None)
        if data is None:
            # This can happen if the pointer was already freed during sleep
            # or if there's a mismatch in the allocator state
            logger.warning(f"Pointer {ptr} not found in allocation registry")
            # Return a dummy handle to prevent crash
            return (0, 0, ptr, 0)
        
        # Clean up CPU backup if it exists
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1], data.tag, ptr)
        return data.handle

    def sleep(
        self,
        offload_tags: Optional[Union[tuple[str, ...], str]] = None
        ) -> None:
        """
        Puts allocations with specific tags to sleep by offloading them to CPU.
        Other active allocations are left untouched.

        :param offload_tags: The tags of the memory allocations that will be
            offloaded.
        """
        logger.info(f"Starting sleep with offload_tags: {offload_tags}")
        # Your debug logging of the allocator state is very helpful, keep it.
        # logger.info(f"Current allocator state before sleep:")
        # logger.info(f" - Total tracked pointers: {len(self.pointer_to_data)}")
        # for i, (ptr, data) in enumerate(list(self.pointer_to_data.items())):
        #     logger.info(f"   {i+1}. Ptr: {ptr}, Tag: {data.tag}, Size: {data.handle[1]}")

        self._is_sleeping = True
        self._freed_pointers.clear()

        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        # Only select pointers that match the specific tags we want to put to sleep.
        # Do not touch any other pointers in the active registry.
        pointers_to_offload = {
            ptr: data for ptr, data in self.pointer_to_data.items()
            if data.tag in offload_tags
        }

        items_to_process = list(pointers_to_offload.items())
        logger.info(f"Found {len(items_to_process)} allocations to offload.")

        # Loop ONLY over the selected pointers for the target model.
        for ptr, data in items_to_process:
            # 1. Remove the pointer from the active registry.
            self.pointer_to_data.pop(ptr)
            
            handle = data.handle
            total_bytes += handle[1]
            backup_bytes += handle[1]
            
            # 2. Offload to CPU.
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device='cpu',
                pin_memory=is_pin_memory_available())
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
            data.cpu_backup_tensor = cpu_backup_tensor
            
            # 3. Add to the sleeping registry.
            self.sleeping_pointer_to_data[ptr] = data
            
            # 4. Unmap from GPU. This is the first and only time for this ptr.
            unmap_and_release(handle)
            # don't need this, memory operations are targeted now
            # self._freed_pointers.add(ptr)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory for the specified tags. "
            "All %.2f GiB is backed up in CPU.",
            total_bytes / 1024**3, backup_bytes / 1024**3
        )

        self._is_sleeping = False
        
        # 400ms bottleneck for 3.5 gb model
        # Simple test runs fine without this
        gc.collect()
      
        torch.cuda.empty_cache()

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        logger.info(f"Starting wake_up with tags: {tags}")
        
        restored_count = 0
        restored_bytes = 0
        
        # Iterate over a copy of the sleeping pointers
        items_to_restore = list(self.sleeping_pointer_to_data.items())

        for ptr, data in items_to_restore:
            # we only restore the pointers for one specific model
            # if tags is None might not be necessary because tags should be always set
            if tags is None or data.tag in tags:
                # First, remove the pointer from the sleeping registry.
                self.sleeping_pointer_to_data.pop(ptr)
                
                handle = data.handle
                logger.debug(f"Restoring ptr {ptr} with tag {data.tag}")
                
                # Re-map the GPU memory.
                create_and_map(handle)
                
                # If there's a CPU backup, copy it to the newly mapped GPU memory.
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    size_in_bytes = cpu_backup_tensor.numel(
                    ) * cpu_backup_tensor.element_size()
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                    data.cpu_backup_tensor = None  # Free the CPU backup
                    restored_count += 1
                    restored_bytes += size_in_bytes
                    logger.debug(f"Restored {size_in_bytes} bytes for ptr {ptr}")

                # Add the now-awake pointer back to the ACTIVE registry.
                self.pointer_to_data[ptr] = data

        # Clear the freed pointers set and exit sleep mode.
        # This should probably only happen when the wake-up is for a specific purpose
        # and not globally. We can leave it for now.
        # self._freed_pointers.clear()
        self._is_sleeping = False
        
        logger.info(f"Wake up complete: restored {restored_count} allocations, "
                    f"{restored_bytes / 1024**3:.2f} GiB")

    @contextmanager
    def use_memory_pool(self, tag: Optional[str] = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        logger.debug(f"Using memory pool with tag: {tag}")
        
        with use_memory_pool_with_allocator(self.python_malloc_callback,
                                            self.python_free_callback) as data:
            # start to hit another PyTorch bug in PyTorch 2.6,
            # possibly because of gc-related issue w.r.t. the allocator and
            # the memory pool.
            # to avoid the issue, we keep a reference of the data.
            # see https://github.com/pytorch/pytorch/issues/146431 .
            self.allocator_and_pools[tag] = data
            yield
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released.
            # right now it is fine, because we only use this allocator
            # during weight loading and kv cache creation, where we only
            # allocate memory.
            # TODO: we need to find a way to release the memory,
            # i.e. calling torch.cuda.empty_cache()
            allocations = data[0].snapshot()
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    # Only free if we're not in sleep mode and pointer wasn't already freed
                    if not self._is_sleeping and allocation["address"] not in self._freed_pointers:
                        handle = self.python_free_callback(allocation["address"])
                        if handle != (0, 0, allocation["address"], 0):  # Not a dummy handle
                            unmap_and_release(handle)
            self.current_tag = old_tag
    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
