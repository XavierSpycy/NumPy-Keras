from ..backend import xp as np

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
        self.__input_shape = None   # (..., C), set when the model is built

    def set_input_shape(
            self,
            shape,
        ) -> None:

        """
        Set the input shape. Batch normalization is per-channel, so gamma
        and beta have the size of the last (channel) axis.

        Parameters:
        - shape (tuple): The input shape without the batch axis.
        """

        self.__input_shape = tuple(shape)

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

        # per-channel parameters: for a 2D input (N, D) the "channels" are
        # the D features; for a 4D input (N, H, W, C) they are the C channels
        n_features = input_dim
        if self.__input_shape is not None and len(self.__input_shape) > 1:
            n_features = self.__input_shape[-1]

        self.params['gamma'] = self.__initializer[self.__gamma_initializer]()((n_features,))
        self.params['beta'] = self.__initializer[self.__beta_initializer]()((n_features,))

        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])

        self.moving_mean = self.__initializer[self.__moving_mean_initializer]()((n_features,))
        self.moving_variance = self.__initializer[self.__moving_variance_initializer]()((n_features,))
        
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
        
        # mean and variance over the batch and all spatial axes, leaving the
        # per-channel axis last: axis 0 for 2D inputs, (0, 1, 2) for 4D
        reduce_axis = tuple(range(inputs.ndim - 1))

        # If the layer is in training mode, compute the outputs using batch normalization
        if is_training:
            batch_mean = np.mean(inputs, axis=reduce_axis)
            batch_var = np.var(inputs, axis=reduce_axis)
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
        
        N = delta.shape[0]
        reduce_axis = tuple(range(delta.ndim - 1))
        # Compute the gradients of weights and biases
        self.grads['gamma'] = np.sum(delta * self.x_normalized, axis=reduce_axis)
        self.grads['beta'] = np.sum(delta, axis=reduce_axis)
        # Normalize the delta
        dx_normalized = delta * self.params['gamma']
        # Compute the delta of mean and variance
        dvar = np.sum(dx_normalized * self.xmu * -0.5 * np.power(self.ivar, 3), axis=reduce_axis)
        dmean = np.sum(dx_normalized * -self.ivar, axis=reduce_axis) + dvar * np.mean(-2. * self.xmu, axis=reduce_axis)
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

    @property
    def output_shape(self):
        return self.__input_shape
    
    def __str__(self):
        return f"BatchNormalization(momentum={self.__momentum}, epsilon={self.__epsilon})"