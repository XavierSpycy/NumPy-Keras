import numpy as np

from ..initializers._mapper import _InitializerMapper

class BatchNormalization(object):
    """
    Batch normalization layer
    """
    def __init__(
            self, 
            momentum: float = 0.9, 
            epsilon: float = 1e-5,
            beta_initializer: str = 'zeros',
            gamma_initializer: str = 'ones',
            moving_mean_initializer: str = 'zeros',
            moving_variance_initializer: str = 'ones',
        ) -> None:

        """
        Initialize the BatchNorm layer.

        Parameters:
        - n_features (int): The number of features.
        - momentum (float): The momentum of the moving average.
        - epsilon (float): The epsilon value.
        """

        self.__momentum = momentum
        self.__epsilon = epsilon
        self.__beta_initializer = beta_initializer
        self.__gamma_initializer = gamma_initializer
        self.__moving_mean_initializer = moving_mean_initializer
        self.__moving_variance_initializer = moving_variance_initializer

        self.__initializer = _InitializerMapper()

    def init_params(
            self, 
            input_dim: int,
        ) -> None:

        """
        Initialize the weights and biases.

        Parameters:
        - input_dim (int): The input dimension.
        """

        self.params = {}
        self.grads = {}

        self.params['gamma'] = self.__initializer[self.__gamma_initializer]()((input_dim,))
        self.params['beta'] = self.__initializer[self.__beta_initializer]()((input_dim,))

        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])

        self.moving_mean = self.__initializer[self.__moving_mean_initializer]()((input_dim,))
        self.moving_variance = self.__initializer[self.__moving_variance_initializer]()((input_dim,))
        
        self.__output_dim = input_dim
    
    def forward(
            self, 
            inputs: np.ndarray, 
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """
        
        # If the layer is in training mode, compute the outputs using batch normalization
        if is_training:
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.xmu = inputs - batch_mean
            self.ivar = 1. / np.sqrt(batch_var + self.epsilon)
            self.x_normalized = self.xmu * self.ivar
            out = self.params['gamma'] * self.x_normalized + self.params['beta']
            self.moving_mean = self.momentum * self.moving_mean + (1. - self.momentum) * batch_mean
            self.moving_variance = self.momentum * self.moving_variance + (1. - self.momentum) * batch_var
        # Otherwise, compute the outputs using the running mean and variance
        else:
            xmu = inputs - self.moving_mean
            ivar = 1. / np.sqrt(self.moving_variance + self.epsilon)
            x_normalized = xmu * ivar
            out = self.params['gamma'] * x_normalized + self.params['beta']
        return out

    def backward(
            self, 
            delta: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation.
        
        Parameters:
        - delta (np.ndarray): The delta of the layer.
        """
        
        N, _ = delta.shape
        # Compute the gradients of weights and biases
        self.grads['gamma'] = np.sum(delta * self.x_normalized, axis=0)
        self.grads['beta'] = np.sum(delta, axis=0)
        # Normalize the delta
        dx_normalized = delta * self.params['gamma']
        # Compute the delta of mean and variance
        dvar = np.sum(dx_normalized * self.xmu * -0.5 * np.power(self.ivar, 3), axis=0)
        dmean = np.sum(dx_normalized * -self.ivar, axis=0) + dvar * np.mean(-2. * self.xmu, axis=0)
        dx = dx_normalized * self.ivar + dvar * 2. * self.xmu / N + dmean / N
        return dx
    
    @property
    def momentum(self):
        return self.__momentum
    
    @property
    def epsilon(self):
        return self.__epsilon
    
    @property
    def output_dim(self):
        return self.__output_dim
    
    def __str__(self):
        return f"BatchNormalization(momentum={self.__momentum}, epsilon={self.__epsilon})"