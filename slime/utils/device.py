import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible device selection.

    Returns the appropriate environment variable name based on the available
    accelerator type (CUDA or Ascend NPU).

    Returns:
        str: 'CUDA_VISIBLE_DEVICES' if CUDA is available,
            'ASCEND_RT_VISIBLE_DEVICES' otherwise.
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Detects the available accelerator and returns the corresponding PyTorch
    device type string. Currently supports CUDA, Ascend NPU, and CPU.

    Returns:
        str: Device type string ('cuda', 'npu', or 'cpu').
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device():
    """Get the PyTorch device module for the current accelerator.

    Returns the torch device namespace (e.g., torch.cuda, torch.npu) based on
    the detected accelerator type. Falls back to torch.cuda if the namespace
    is not found.

    Returns:
        module: The PyTorch device module (torch.cuda, torch.npu, etc.).
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Get the index of the current accelerator device.

    Returns:
        int: The current device index (e.g., 0 for 'cuda:0').
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Get the distributed communication backend based on device type.

    Returns the appropriate collective communication backend for the
    detected accelerator (HCCL for Ascend NPU, NCCL for CUDA).

    Returns:
        str: Backend name ('hccl' for NPU, 'nccl' for CUDA/default).
    """
    if is_npu_available:
        return "hccl"
    else:
        # default to nccl
        return "nccl"
