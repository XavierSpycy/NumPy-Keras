from . import functional as F

class _ActivationMapper:
    activations = {
        "linear": F.linear, 
        "elu": F.elu, 
        "hardshrink": F.hardshrink, 
        "hardsigmoid": F.hardsigmoid, 
        "hardtanh": F.hardtanh, 
        "hardswish": F.hardswish,
        "leaky_relu": F.leaky_relu, 
        "log_sigmoid": F.log_sigmoid, 
        "relu": F.relu, 
        "relu6": F.relu6, 
        "selu": F.selu, 
        "celu": F.celu, 
        "gelu": F.gelu,
        "sigmoid": F.sigmoid, 
        "silu": F.silu,
        "mish": F.mish,
        "softplus": F.softplus, 
        "softshrink": F.softshrink, 
        "softsign": F.softsign, 
        "tanh": F.tanh, 
        "tanhshrink": F.tanhshrink,
        "threshold": F.threshold,
        "softmax": F.softmax, 
    }

    def __getitem__(self, name: str):
        if not name.lower() in self.activations:
            raise ValueError(f"Activation function {name} not found.")
        else:
            return self.activations[name.lower()]