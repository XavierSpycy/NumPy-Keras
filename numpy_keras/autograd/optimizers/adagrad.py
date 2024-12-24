from typing import Dict

import numpy as np

from ._base import Optimizer

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
            params: Dict,
            grads: Dict,
        ) -> None:
        """
        Update rule of Adagrad for the parameters of the model.

        Parameters:
        - params (Dict): A dictionary of parameters.
        - grads (Dict): A dictionary of gradients.
        """
        # Initialize the grad_square
        if not self.grad_square:
            self.init_grad_square(params)
        
        for key in params:
            # Get the gradient
            grad = grads[key]
            # Add weight decay
            grad += self.weight_decay * params[key]
            # Update the grad_square
            self.grad_square[key] += np.square(grad)
            # Update the parameters
            params[key] -= self.learning_rate * grad / (np.sqrt(self.grad_square[key]) + self.epsilon)
        
    def init_grad_square(
            self, 
            params: Dict,
        ) -> None:

        """
        Initialize the grad_square.

        Parameters:
        - params (Dict): A dictionary of parameters.
        """

        for param in params:
            self.grad_square[param] = np.zeros_like(params[param])