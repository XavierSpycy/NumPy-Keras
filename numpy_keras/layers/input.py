import math
from typing import Tuple, Union

class Input:
    """
    The Input layer.
    """
    def __init__(
            self,
            shape: Union[int, Tuple[int, ...]],
        ) -> None:

        """
        Initialize the Input layer.

        Parameters:
        - shape (int or tuple): The shape of a single sample, without the
          batch axis. An int means a 1D feature vector of that size; a tuple
          gives the full shape, e.g. (28, 28, 1) for a 28x28 grayscale image.
        """

        if isinstance(shape, int):
            shape = (shape,)
        elif not (isinstance(shape, (tuple, list))
                  and shape
                  and all(isinstance(dim, int) for dim in shape)):
            raise ValueError("Input shape must be an int or a tuple of ints.")

        self.__shape = tuple(shape)
        self.__output_dim = math.prod(self.__shape)

    @property
    def output_dim(self):
        return self.__output_dim

    @property
    def output_shape(self):
        return self.__shape

    def __str__(self):
        if len(self.__shape) == 1:
            return f"Input(shape={self.__shape[0]})"
        return f"Input(shape={self.__shape})"
