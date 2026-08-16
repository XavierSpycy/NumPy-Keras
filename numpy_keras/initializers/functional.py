from typing import Tuple

import numpy as _np  # host RNG + scalar math: seeded runs stay bit-identical across backends
from ..backend import xp as np

def uniform(shape: Tuple[int, int], a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return np.asarray(_np.random.uniform(low=a, high=b, size=shape))

def normal(shape: Tuple[int, int], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.asarray(_np.random.normal(loc=mean, scale=std, size=shape))

def constant(shape: Tuple[int, int], value: float = 0.0) -> np.ndarray:
    return np.full(shape, value)

def ones(shape: Tuple[int, int]) -> np.ndarray:
    return np.ones(shape)

def zeros(shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros(shape)

def _fan_in(shape: Tuple[int, ...]) -> int:
    """Number of inputs a neuron receives: all axes but the last.

    For a Dense kernel (in, out) this is `in`; for a Conv2D kernel
    (kh, kw, in_channels, filters) it is kh * kw * in_channels."""
    return int(_np.prod(shape[:-1]))

def _fan_out(shape: Tuple[int, ...]) -> int:
    """Number of outputs a neuron feeds: all axes but the first."""
    return int(_np.prod(shape[1:]))

def xaiver_uniform(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    return np.asarray(gain * _np.random.uniform(low=-_np.sqrt(6 / (_fan_in(shape) + _fan_out(shape))), high=_np.sqrt(6 / (_fan_in(shape) + _fan_out(shape))), size=shape))

def xaiver_normal(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    return np.asarray(gain * _np.random.normal(loc=0.0, scale=_np.sqrt(2 / (_fan_in(shape) + _fan_out(shape))), size=shape))

def kaiming_uniform(shape: Tuple[int, int], mode: str = 'fan_in') -> np.ndarray:
    fan = _fan_in(shape) if mode == 'fan_in' else _fan_out(shape)
    return np.asarray(_np.random.uniform(low=-_np.sqrt(6/fan), high=_np.sqrt(6/fan), size=shape))

def kaiming_normal(shape: Tuple[int, int], mode: str = 'fan_in') -> np.ndarray:
    fan = _fan_in(shape) if mode == 'fan_in' else _fan_out(shape)
    return np.asarray(_np.random.normal(loc=0.0, scale=_np.sqrt(2/fan), size=shape))
