try:
    import autograd.numpy as np
except:
    pass

def linear(x):
    return x

def elu(x, alpha: float = 1.0):
    x = np.clip(x, -709, 709)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def hardshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x, np.where(x < -lambd, x, 0))

def hardsigmoid(x):
    return np.clip(x * 1/6 + 1/2, 0, 1)

def hardtanh(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)

def hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6

def leaky_relu(x, alpha: float = 0.01):
    return np.where(x >= 0, x, alpha * x)

def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

def relu6(x):
    return np.clip(x, 0, 6)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.clip(x, -709, 709)
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def celu(x, alpha: float = 1.0):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def sigmoid(x):
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-x[positive]))
    negative = ~positive
    exp_x = np.exp(x[negative])
    sigmoid[negative] = exp_x / (1 + exp_x)
    return sigmoid

def silu(x):
    return x * sigmoid(x)

def mish(x):
    return x * tanh(softplus(x))

def softplus(x, beta: float = 1.0, threshold: float = 20.0):
    return np.where(x * beta > threshold, x, np.log(1 + np.exp(beta * x)) / beta)

def softshrink(x, lambd: float = 0.5):
    return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))

def softsign(x):
    return x / (1 + np.abs(x))

def tanh(x):
    return np.tanh(x)

def tanhshrink(x):
    return x - np.tanh(x)

def threshold(x, threshold: float, value: float):
    return np.where(x > threshold, x, value)

def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)