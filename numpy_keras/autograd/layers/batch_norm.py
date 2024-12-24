try:
    import autograd.numpy as np
except:
    pass

from ..initializers._mapper import _InitializerMapper
from ...initializers._mapper import _InitializerMapper as _PureInitializerMapper

class BatchNormalization:
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
        - momentum (float): The momentum of the moving average.
        - epsilon (float): The epsilon value to prevent division by zero.
        - beta_initializer (str): Initializer for beta.
        - gamma_initializer (str): Initializer for gamma.
        - moving_mean_initializer (str): Initializer for moving mean.
        - moving_variance_initializer (str): Initializer for moving variance.
        """

        self.__momentum = momentum
        self.__epsilon = epsilon
        self.__beta_initializer = beta_initializer
        self.__gamma_initializer = gamma_initializer
        self.__moving_mean_initializer = moving_mean_initializer
        self.__moving_variance_initializer = moving_variance_initializer

        self.__initializer = _InitializerMapper()
        self.__pure_initializer = _PureInitializerMapper()

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

        # Initialize gamma and beta with autograd.numpy initializers
        self.params['gamma'] = self.__initializer[self.__gamma_initializer]()((input_dim,))
        self.params['beta'] = self.__initializer[self.__beta_initializer]()((input_dim,))

        # Initialize gradients for gamma and beta
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])

        # Initialize moving mean and moving variance with pure numpy
        self.moving_mean = self.__pure_initializer[self.__moving_mean_initializer]()((input_dim,))
        self.moving_variance = self.__pure_initializer[self.__moving_variance_initializer]()((input_dim,))
        
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
        - is_training (bool): Whether in training mode.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """

        if is_training:
            # Compute batch statistics
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            # Convert batch_mean and batch_var to pure numpy arrays to update moving stats
            batch_mean_val = np.array(list(batch_mean))
            batch_var_val = np.array(list(batch_var))
            # Update moving mean and variance with pure numpy
            self.moving_mean = self.__momentum * self.moving_mean + (1. - self.__momentum) * batch_mean_val
            self.moving_variance = self.__momentum * self.moving_variance + (1. - self.__momentum) * batch_var_val
            # Normalize
            xmu = inputs - batch_mean
            ivar = 1. / np.sqrt(batch_var + self.__epsilon)
            x_normalized = xmu * ivar
            # Scale and shift
            out = self.params['gamma'] * x_normalized + self.params['beta']
        else:
            # Use moving statistics
            xmu = inputs - self.moving_mean
            ivar = 1. / np.sqrt(self.moving_variance + self.__epsilon)
            x_normalized = xmu * ivar
            out = self.params['gamma'] * x_normalized + self.params['beta']
        
        return out

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