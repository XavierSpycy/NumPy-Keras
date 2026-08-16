from typing import List

from ..backend import xp as np, is_cupy_array, is_numpy_array
from . import _gpu_kernels as _gk

from ._base import Optimizer
from ..cython import _kernels as _ck

class Adagrad(Optimizer):
    """
    Adagrad optimizer.
    """
    def __init__(
            self, 
            learning_rate: float = 1.0, 
            weight_decay: float = 0.0, 
            epsilon: float = 1e-10,
        ) -> None:

        """
        Initialize the Adagrad optimizer.

        Parameters:
        - learning_rate (float, optional): Learning rate. Default is 1.0.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-10.
        """

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.grad_square = {}
        
    def update(
            self, 
            layers: List,
        ) -> None:

        """
        Update rule of Adagrad for the parameters of the model.

        Parameters:
        - layers (List): A list of layers in the model.
        """

        # Initialize the grad_square
        if not self.grad_square:
            self.init_grad_square(layers)
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Optional Cython fast path: one fused pass per param array.
                    # The kernels are CPU-only, so device arrays take the pure
                    # path; the type checks must come before any .flags access
                    # (cupy arrays have no .flags attribute).
                    if (_ck is not None
                            and is_numpy_array(layer.params[key])
                            and is_numpy_array(layer.grads[key])
                            and is_numpy_array(self.grad_square[i][key])
                            and layer.params[key].dtype == np.float64
                            and layer.params[key].flags.c_contiguous
                            and layer.grads[key].flags.c_contiguous
                            and self.grad_square[i][key].flags.c_contiguous):
                        _ck.adagrad_update(
                            layer.params[key], layer.grads[key],
                            self.grad_square[i][key],
                            self.learning_rate, self.epsilon, self.weight_decay)
                        continue
                    # Optional GPU fast path: one fused pass per param array
                    # (the device twin of the Cython kernels above)
                    if (is_cupy_array(layer.params[key])
                            and layer.params[key].dtype in (np.float32, np.float64)):
                        _gk.adagrad_update(
                            layer.params[key], layer.grads[key],
                            self.grad_square[i][key],
                            self.learning_rate, self.epsilon, self.weight_decay)
                        continue
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update the grad_square
                    self.grad_square[i][key] += np.square(grad)
                    # Update the parameters
                    layer.params[key] -= self.learning_rate * grad / (np.sqrt(self.grad_square[i][key]) + self.epsilon)
    
    def init_grad_square(
            self, 
            layers: List,
        ) -> None:

        """
        Initialize the grad_square.

        Parameters:
        - layers (List): A list of layers in the model.
        """

        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.grad_square[i] = {}
                for key in layer.params:
                    self.grad_square[i][key] = np.zeros_like(layer.params[key])