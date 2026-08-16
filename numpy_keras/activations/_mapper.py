from functools import lru_cache

from ..backend import xp as np, is_cupy_array

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
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _lookup(name: str):
        return _ActivationMapper.activations[name.lower()]

    def __getitem__(self, name: str):
        try:
            return self._lookup(name)
        except KeyError:
            raise ValueError(f"Activation function {name} not found.")

    def backward(
            self,
            name: str,
            output: np.ndarray,
            grad: np.ndarray,
            config: dict,
        ) -> np.ndarray:
        """Chain a gradient through this activation: returns grad ⊙ f'(output).

        Each layer owns its own activation, so backward is a local operation
        evaluated on the layer's cached post-activation output.  softmax has
        no elementwise derivative: its Jacobian J[i, j] = ŷ_i (δ_ij - ŷ_j) is
        contracted with the gradient along the last (class) axis.  linear /
        None pass the gradient through unchanged.
        """
        if name is None or name == "linear":
            return grad
        if name == "softmax":
            y = output.reshape(-1, output.shape[-1])
            g = grad.reshape(-1, grad.shape[-1])
            if is_cupy_array(output):
                # contracted form y ⊙ (g - g·y) of the same
                # Jacobian-vector product: mathematically identical
                # (dz_i = y_i (g_i - Σ_j y_j g_j)), but it does not
                # materialize the (n, C, C) Jacobian and avoids cupy's
                # einsum overhead (which dominates at small batch sizes)
                dz = y * (g - (g * y).sum(axis=1, keepdims=True))
                return dz.reshape(grad.shape)
            J = y[:, :, None] * (np.eye(y.shape[-1])[None] - y[:, None, :])
            dz = np.einsum("nij,nj->ni", J, g)
            return dz.reshape(grad.shape)
        deriv = self[name + "_deriv"]
        return grad * deriv(output, **config)