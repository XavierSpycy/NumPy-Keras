from typing import Tuple

import numpy as np

def uniform(shape: Tuple[int, int], a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return np.random.uniform(low=a, high=b, size=shape)

def normal(shape: Tuple[int, int], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.random.normal(loc=mean, scale=std, size=shape)

def constant(shape: Tuple[int, int], value: float = 0.0) -> np.ndarray:
    return np.full(shape, value)

def ones(shape: Tuple[int, int]) -> np.ndarray:
    return np.ones(shape)

def zeros(shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros(shape)

def xaiver_uniform(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    return gain * np.random.uniform(low=-np.sqrt(6 / (shape[0] + shape[1])), high=np.sqrt(6 / (shape[0] + shape[1])), size=shape)

def xaiver_normal(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    return gain * np.random.normal(loc=0.0, scale=np.sqrt(2 / (shape[0] + shape[1])), size=shape)

def kaiming_uniform(shape: Tuple[int, int], mode: str = 'fan_in') -> np.ndarray:
    return np.random.uniform(low=-np.sqrt(3/(shape[0] if mode == 'fan_in' else shape[1])), high=np.sqrt(3/(shape[0] if mode == 'fan_in' else shape[1])), size=shape)

def kaiming_normal(shape: Tuple[int, int], mode: str = 'fan_in') -> np.ndarray:
    return np.random.normal(loc=0.0, scale=np.sqrt(1/(shape[0] if mode == 'fan_in' else shape[1])), size=shape)