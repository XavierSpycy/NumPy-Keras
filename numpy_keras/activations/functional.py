from ..backend import xp as np


def _same_dtype_as_input(fn):
    """Cast an elementwise result back to the input's dtype.

    All functions in this module are f(x) / f_deriv(a) elementwise ops,
    so their output dtype should always follow the input.  Without this,
    Python scalar arguments (``np.clip`` bounds, ``np.where`` branches,
    ``np.maximum``'s floor) reach cupy as 0-d float64 arrays and silently
    promote a float32 model to float64.  The cast is a no-op whenever the
    result already has the right dtype (numpy backend, float64 models).
    """
    def wrapper(x, *args, **kwargs):
        out = fn(x, *args, **kwargs)
        arr = np.asarray(out)
        if arr.dtype != x.dtype:
            return np.asarray(out, dtype=x.dtype)
        return out
    return wrapper


@_same_dtype_as_input
def linear(x):
    return x

@_same_dtype_as_input
def linear_deriv(a):
    return 1

@_same_dtype_as_input
def elu(x, alpha: float = 1.0):
    x = np.clip(x, -709, 709)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

@_same_dtype_as_input
def elu_deriv(a, alpha: float = 1.0):
    return np.where(a > 0, 1, a + alpha)

@_same_dtype_as_input
def hardshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x, np.where(x < -lambd, x, 0))

@_same_dtype_as_input
def hardshrink_deriv(a, lambd: float = 0.5):
    return np.where((a > lambd) | (a < -lambd), 1, 0)

@_same_dtype_as_input
def hardsigmoid(x):
    return np.clip(x * 1/6 + 1/2, 0, 1)

@_same_dtype_as_input
def hardsigmoid_deriv(a):
    return np.where((a > 0) & (a < 1), 1/6, 0)

@_same_dtype_as_input
def hardtanh(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)

@_same_dtype_as_input
def hardtanh_deriv(a, min_val=-1.0, max_val=1.0):
    return np.where((a > min_val) & (a < max_val), 1, 0)

@_same_dtype_as_input
def leaky_relu(x, alpha: float = 0.01):
    return np.where(x >= 0, x, alpha * x)

@_same_dtype_as_input
def leaky_relu_deriv(a, alpha: float = 0.01):
    return np.where(a >= 0, 1, alpha)

@_same_dtype_as_input
def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))

@_same_dtype_as_input
def log_sigmoid_deriv(a):
    return 1 - np.exp(a)

@_same_dtype_as_input
def relu(x):
    return x * (x > 0)

@_same_dtype_as_input
def relu_deriv(a):
    return 1. * (a > 0)

@_same_dtype_as_input
def relu6(x):
    return np.clip(x, 0, 6)

@_same_dtype_as_input
def relu6_deriv(a):
    return np.where((a > 0) & (a < 6), 1, 0)

@_same_dtype_as_input
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.clip(x, -709, 709)
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

@_same_dtype_as_input
def selu_deriv(a):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.where(a > 0, a / scale, np.log(np.maximum(a / (scale * alpha) + 1, 1e-10)))
    x = np.clip(x, -709, 709)
    return np.where(a > 0, scale, scale * alpha * np.exp(x))

@_same_dtype_as_input
def celu(x, alpha: float = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

@_same_dtype_as_input
def celu_deriv(a, alpha: float = 1.0):
    x = np.where(a > 0, 1, np.log(np.maximum((a + alpha) / alpha, 1e-10)))
    x = np.clip(x, -709, 709)
    return np.where(a > 0, 1, np.exp(x))

@_same_dtype_as_input
def sigmoid(x):
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-x[positive]))
    negative = ~positive
    exp_x = np.exp(x[negative])
    sigmoid[negative] = exp_x / (1 + exp_x)
    return sigmoid

@_same_dtype_as_input
def sigmoid_deriv(a):
    return a * (1 - a)

@_same_dtype_as_input
def softplus(x, beta: float = 1.0, threshold: float = 20.0):
    return np.where(x * beta > threshold, x, np.log(1 + np.exp(beta * x)) / beta)

@_same_dtype_as_input
def softplus_deriv(a, beta: float = 1.0, threshold: float = 20.0):
    return np.where(a > threshold, 1, 1 - np.exp(-beta * a))

@_same_dtype_as_input
def softshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))

@_same_dtype_as_input
def softshrink_deriv(a, lambd: float = 0.5):
    return np.where(a != 0, 1, 0)

@_same_dtype_as_input
def softsign(x):
    return x / (1 + np.abs(x))

@_same_dtype_as_input
def softsign_deriv(a):
    return np.where(a >= 0, (1 - a) ** 2, (1 + a) ** 2)

@_same_dtype_as_input
def tanh(x):
    return np.tanh(x)

@_same_dtype_as_input
def tanh_deriv(a):
    return 1 - a ** 2

@_same_dtype_as_input
def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)