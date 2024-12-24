import numpy as np

def linear(x):
    return x

def linear_deriv(a):
    return 1

def elu(x, alpha: float = 1.0):
    x = np.clip(x, -709, 709)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(a, alpha: float = 1.0):
    return np.where(a > 0, 1, a + alpha)

def hardshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x, np.where(x < -lambd, x, 0))

def hardshrink_deriv(a, lambd: float = 0.5):
    return np.where((a > lambd) | (a < -lambd), 1, 0)

def hardsigmoid(x):
    return np.clip(x * 1/6 + 1/2, 0, 1)

def hardsigmoid_deriv(a):
    return np.where((a == 0) & (a == 1), 1/6, 0)

def hardtanh(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)

def hardtanh_deriv(a, min_val=-1.0, max_val=1.0):
    return np.where((a >= min_val) & (a <= max_val), 1, 0)

def leaky_relu(x, alpha: float = 0.01):
    return np.where(x >= 0, x, alpha * x)

def leaky_relu_deriv(a, alpha: float = 0.01):
    return np.where(a >= 0, 1, alpha)

def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))

def log_sigmoid_deriv(a):
    a_exp = np.exp(a)
    return a_exp * (1 - a_exp)

def relu(x):
    return x * (x > 0)

def relu_deriv(a):
    return 1. * (a > 0)

def relu6(x):
    return np.clip(x, 0, 6)

def relu6_deriv(a):
    return np.where((a >= 0) & (a <= 6), 1, 0)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.clip(x, -709, 709)
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_deriv(a):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.where(a > 0, a / scale, np.log(np.maximum(a / (scale * alpha), 1e-10) + 1))
    x = np.clip(x, -709, 709)
    return np.where(a > 0, scale, scale * alpha * np.exp(x))

def celu(x, alpha: float = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

def celu_deriv(a, alpha: float = 1.0):
    x = np.where(a > 0, 1, np.log(np.maximum((a + 1) / alpha, 1e-10) * alpha))
    x = np.clip(x, -709, 709)
    return np.where(a > 0, 1, alpha * np.exp(x))

def sigmoid(x):
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-x[positive]))
    negative = ~positive
    exp_x = np.exp(x[negative])
    sigmoid[negative] = exp_x / (1 + exp_x)
    return sigmoid

def sigmoid_deriv(a):
    return a * (1 - a)

def softplus(x, beta: float = 1.0, threshold: float = 20.0):
    return np.where(x * beta > threshold, x, np.log(1 + np.exp(beta * x)) / beta)

def softplus_deriv(a, beta: float = 1.0, threshold: float = 20.0):
    return np.where(a > threshold, 1, 1 / (1 + np.exp(-beta * a)))

def softshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))

def softshrink_deriv(a, lambd: float = 0.5):
    return np.where(a != 0, 1, 0)

def softsign(x):
    return x / (1 + np.abs(x))

def softsign_deriv(a):
    return np.where(a >= 0, (1 - a) ** 2, (1 + a) ** 2)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(a):
    return 1 - a ** 2

def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)