from typing import List

import numpy as np

from ._base import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(
            self, 
            learning_rate: float = 1e-3, 
            beta1: float = 0.9, 
            beta2: float = 0.999, 
            epsilon: float = 1e-08, 
            weight_decay: float = 0.0,
        ) -> None:

        """
        Initialize the Adam optimizer.

        Parameters:
        - learning_rate (float, optional): Learning rate. Default is 1e-3.
        - beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        - beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-8.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.first_moment = {}
        self.second_moment = {}
        self.t = 1
        
    def update(
            self, 
            layers: List,
        ) -> None:
        """
        Update rule of Adam for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        if not self.first_moment:
            self.init_moment(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update biased first moment estimate
                    self.first_moment[i][key] *= self.beta1
                    # Update biased second raw moment estimate
                    self.second_moment[i][key] *= self.beta2
                    # Correct bias
                    self.first_moment[i][key] += (1 - self.beta1) * layer.grads[key]
                    self.second_moment[i][key] += (1 - self.beta2) * np.square(layer.grads[key])
                    # Update parameters
                    first_moment_hat = self.first_moment[i][key] / (1 - self.beta1 ** self.t)
                    second_moment_hat = self.second_moment[i][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + self.epsilon)
        self.t += 1
        
    def init_moment(
            self, 
            layers: List,
        ) -> None:

        """
        Initialize the first_moment and second_moment.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.first_moment[i] = {}
                self.second_moment[i] = {}
                for key in layer.params:
                    self.first_moment[i][key] = np.zeros_like(layer.params[key])
                    self.second_moment[i][key] = np.zeros_like(layer.params[key])