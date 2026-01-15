import gc
import logging

import torch
import torch.distributed as dist
from slime.utils.device import get_torch_device

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    get_torch_device().synchronize()
    gc.collect()
    get_torch_device().empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = get_torch_device().current_device()
    free, total = get_torch_device().mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(get_torch_device().memory_allocated(device)),
        "reserved_GB": _byte_to_gb(get_torch_device().memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
