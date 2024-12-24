from typing import Dict

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
        self.accum_grad_square = {}
        self.accum_delta_square = {}
    
    def update(
            self, 
            params: Dict,
            grads: Dict,
        ) -> None:

        """
        Update rule of Adadelta for the parameters of the model.

        Parameters:
        - params (Dict): The parameters of the model.
        - grads (Dict): The gradients of the parameters.
        """

        if not self.accum_grad_square:
            self.init_accum_square(params)
        
        for key in params:
            # Get the gradient
            grad = grads[key]
            # Add weight decay
            grad += self.weight_decay * params[key]
            # Update the accum_grad_square
            self.accum_grad_square[key] *= self.rho
            self.accum_grad_square[key] += (1 - self.rho) * np.square(grad)
            # Calculate the delta
            delta = - np.sqrt(self.accum_delta_square[key] + self.epsilon) / np.sqrt(self.accum_grad_square[key] + self.epsilon) * grad
            # Update the parameter
            params[key] += self.learning_rate * delta
            # Update the accum_delta_square
            self.accum_delta_square[key] *= self.rho
            self.accum_delta_square[key] += (1 - self.rho) * np.square(delta)
                    
    def init_accum_square(
            self, 
            params: Dict,
        ) -> None:

        """
        Initialize the accum_grad_square and accum_delta_square.

        Parameters:
        - params (Dict): The parameters of the model.
        """
        
        for key in params:
            self.accum_grad_square[key] = np.zeros_like(params[key])
            self.accum_delta_square[key] = np.zeros_like(params[key])