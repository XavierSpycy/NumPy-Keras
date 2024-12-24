from . import functional as F

class _ActivationMapper:
    activations = {
        "linear": F.linear, "linear_deriv": F.linear_deriv,
        "elu": F.elu, "elu_deriv": F.elu_deriv,
        "hardshrink": F.hardshrink, "hardshrink_deriv": F.hardshrink_deriv,
        "hardsigmoid": F.hardsigmoid, "hardsigmoid_deriv": F.hardsigmoid_deriv,
        "hardtanh": F.hardtanh, "hardtanh_deriv": F.hardtanh_deriv,
        "leaky_relu": F.leaky_relu, "leaky_relu_deriv": F.leaky_relu_deriv,
        "log_sigmoid": F.log_sigmoid, "log_sigmoid_deriv": F.log_sigmoid_deriv,
        "relu": F.relu, "relu_deriv": F.relu_deriv,
        "relu6": F.relu6, "relu6_deriv": F.relu6_deriv,
        "selu": F.selu, "selu_deriv": F.selu_deriv,
        "celu": F.celu, "celu_deriv": F.celu_deriv,
        "sigmoid": F.sigmoid, "sigmoid_deriv": F.sigmoid_deriv,
        "softplus": F.softplus, "softplus_deriv": F.softplus_deriv,
        "softshrink": F.softshrink, "softshrink_deriv": F.softshrink_deriv,
        "softsign": F.softsign, "softsign_deriv": F.softsign_deriv,
        "tanh": F.tanh, "tanh_deriv": F.tanh_deriv,
        "softmax": F.softmax, 
    }
    
    def __getitem__(self, name: str):
        if not name.lower() in self.activations:
            raise ValueError(f"Activation function {name} not found.")
        else:
            return self.activations[name.lower()]