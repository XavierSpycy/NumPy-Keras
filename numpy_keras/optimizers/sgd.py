from typing import List

import numpy as np

from ._base import Optimizer

class SGD(Optimizer):
    """
    SGD optimizer, including SGD with momentum and Nesterov accelerated gradient.
    """
    def __init__(
            self, 
            learning_rate, 
            momentum: float = 0.0, 
            nesterov: bool = False, 
            weight_decay: float = 0.0,
        ) -> None:

        """
        Initialize the SGD optimizer.

        Parameters:
        - learning_rate (float): Learning rate.
        - momentum (float, optional): Momentum factor. Default is 0.0.
        - nesterov (bool, optional): Whether to use Nesterov accelerated gradient (NAG). Default is False.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = {}
        self.velocity_prev = {} if self.nesterov else None
    
    def update(
            self, 
            layers: List,
        ) -> None:
        
        """
        Update rule of SGD for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """

        if not self.velocity:
            self.init_velocity(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # If SGD with Nesterov accelerated gradient (NAG) is used
                    if self.nesterov:
                        # Update the velocity
                        self.velocity_prev[key] = self.velocity[i][key]
                        self.velocity[i][key] = self.momentum * self.velocity_prev[key] - self.learning_rate * grad
                        # Update the parameters
                        layer.params[key] -= -self.momentum * self.velocity_prev[key] - (1 + self.momentum) * self.velocity[i][key]
                    else:
                        # Update the velocity
                        self.velocity[i][key] = self.momentum * self.velocity[i][key] + self.learning_rate * grad
                        # Update the parameters
                        layer.params[key] -= self.velocity[i][key]
    
    def init_velocity(
            self, 
            layers: List,
        ) -> None:
        
        """
        Initialize the velocity.

        Parameters:
        - layers (List): A list of layers in the model.
        """
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.velocity[i] = {}
                for key in layer.params:
                    self.velocity[i][key] = np.zeros_like(layer.params[key])