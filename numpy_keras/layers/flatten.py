import math
from typing import Optional, Tuple

import numpy as np

class Flatten:
    """
    Flatten layer
    """
    def __init__(
            self,
            input_shape: Optional[Tuple[int, ...]] = None,
        ) -> None:

        """
        Initialize the Flatten layer.

        Parameters:
        - input_shape (tuple, optional): The input shape. Default is None,
          in which case it is inferred from the previous layer during model
          construction.
        """

        self.__input_shape = input_shape

    def set_input_shape(
            self,
            shape: Tuple[int, ...],
        ) -> None:

        """
        Set the input shape. Called by Sequential during construction;
        an explicitly passed input_shape always wins.
        """

        if self.__input_shape is None:
            self.__input_shape = shape

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool = True,
        ) -> np.ndarray:

        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.
        - is_training (bool): Whether the model is training.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """

        return inputs.reshape(inputs.shape[0], -1)

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: reshape the gradient back to the input shape.

        Parameters:
        - grad (ndarray): The gradient of the loss w.r.t. this layer's
          output, of shape (N, prod(input_shape)).

        Returns:
        - grad (ndarray): The gradient of the loss w.r.t. the input, of
          shape (N,) + input_shape.
        """

        return grad.reshape((grad.shape[0],) + self.__input_shape)

    @property
    def output_dim(self):
        return math.prod(self.__input_shape)

    @property
    def output_shape(self):
        return (math.prod(self.__input_shape),)

    def __str__(self):
        return f"Flatten(input_shape={self.__input_shape})"
