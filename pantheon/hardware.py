"""Best-effort hardware capability probing.

Detecting a device does not mean PANTHEON can execute simulation kernels on it.
"""
from dataclasses import dataclass
import importlib.util
import platform


@dataclass(frozen=True)
class Device:
    backend: str
    name: str
    available: bool
    status: str


def detect_devices() -> list[Device]:
    devices = [Device("cpu", platform.machine(), True, "reference backend")]
    if importlib.util.find_spec("torch") is None:
        devices += [
            Device("cuda", "NVIDIA CUDA", False, "torch not installed"),
            Device("ascend", "Huawei Ascend", False, "torch not installed"),
        ]
        return devices
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        devices.append(Device("cuda", "NVIDIA CUDA", cuda, "planned runtime adapter"))
    except Exception:
        devices.append(Device("cuda", "NVIDIA CUDA", False, "probe unavailable"))
    try:
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa: F401
            import torch
            available = bool(torch.npu.is_available())
        else:
            available = False
        devices.append(Device("ascend", "Huawei Ascend", available, "planned runtime adapter"))
    except Exception:
        devices.append(Device("ascend", "Huawei Ascend", False, "probe unavailable"))
    return devices
