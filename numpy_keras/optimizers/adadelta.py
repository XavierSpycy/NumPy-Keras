from typing import List

import numpy as np

from ._base import Optimizer

class Adadelta(Optimizer):
    """
    Adadelta optimizer.
    """
    def __init__(
            self, 
            learning_rate: float = 1.0, 
            rho: float = 0.9, 
            epsilon: float = 1e-06, 
            weight_decay: float = 0.0
        ) -> None:
        
        """
        Initialize the Adadelta optimizer.
        
        Parameters:
        - learning_rate (float, optional): Learning rate. Default is 1.0.
        - rho (float, optional): Decay rate. Default is 0.9.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-6.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """
        
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.accum_grad_square = None
        self.accum_delta_square = None
    
    def update(
            self, 
            layers: List
        ) -> None:

        """
        Update rule of Adadelta for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """

        if self.accum_grad_square is None:
            self.init_accum_square(layers)
    
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update the accum_grad_square
                    self.accum_grad_square[i][key] *= self.rho
                    self.accum_grad_square[i][key] += (1 - self.rho) * np.square(layer.grads[key])
                    # Calculate the delta
                    delta = - np.sqrt(self.accum_delta_square[i][key] + self.epsilon) / np.sqrt(self.accum_grad_square[i][key] + self.epsilon) * layer.grads[key]
                    # Update the accum_delta_square
                    layer.params[key] += self.learning_rate * delta
                    # Update the accum_delta_square
                    self.accum_delta_square[i][key] *= self.rho
                    self.accum_delta_square[i][key] += (1 - self.rho) * np.square(delta)
                    
    def init_accum_square(
            self, 
            layers: List,
        ) -> None:

        """
        Initialize the accum_grad_square and accum_delta_square.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        
        self.accum_grad_square = {}
        self.accum_delta_square = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.accum_grad_square[i] = {}
                self.accum_delta_square[i] = {}
                for key in layer.params:
                    self.accum_grad_square[i][key] = np.zeros_like(layer.params[key])
                    self.accum_delta_square[i][key] = np.zeros_like(layer.params[key])