"""
LUMINARK framework package.

Heavy optional PyTorch components are imported lazily so non-neural modules,
including the SAP stage framework, can be used without installing PyTorch.
"""

__all__ = ["Module", "LuminarkTrainer", "LuminarkSafetySystem"]


def __getattr__(name):
    if name == "Module":
        from .nn import Module
        return Module
    if name == "LuminarkTrainer":
        from .training.trainer import LuminarkTrainer
        return LuminarkTrainer
    if name == "LuminarkSafetySystem":
        from .monitoring.defense import LuminarkSafetySystem
        return LuminarkSafetySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
