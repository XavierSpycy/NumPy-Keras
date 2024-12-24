from typing import Dict

try:
    import autograd.numpy as np
except:
    pass

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
            params: Dict, 
            grads: Dict,
        ) -> None:

        """
        Update rule of SGD for the parameters of the model.

        Parameters:
        - params (Dict): A dictionary of parameters.
        - grads (Dict): A dictionary of gradients.
        """
        
        if not self.velocity:
            self.init_velocity(params)
        
        for key in params:
            # Get the gradient
            grad = grads[key]
            # Add weight decay
            grad += self.weight_decay * params[key]
            if self.nesterov:
                # Update the velocity
                self.velocity_prev[key] = self.velocity[key]
                self.velocity[key] = self.momentum * self.velocity_prev[key] - self.learning_rate * grad
                # Update the parameters
                params[key] -= -self.momentum * self.velocity_prev[key] - (1 + self.momentum) * self.velocity[key]
            else:
                # Update the velocity
                self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * grad
                # Update the parameters
                params[key] -= self.velocity[key]
    
    def init_velocity(
            self, 
            params: Dict,
        ) -> None:
        
        """
        Initialize the velocity.
        
        Parameters:
        - params (Dict): A dictionary of parameters.
        """
        
        for key in params:
            self.velocity[key] = np.zeros_like(params[key])