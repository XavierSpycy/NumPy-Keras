import math
from typing import Tuple

import numpy as np

class Flatten:
    """
    Flatten layer
    """
    def __init__(
            self, 
            input_shape: Tuple[int, ...],
        ) -> None:

        """
        Initialize the Flatten layer.

        Parameters:
        - input_shape (tuple): The input shape.
        """

        self.__input_shape = input_shape
    
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
    
    @property
    def output_dim(self):
        return math.prod(self.__input_shape)
    
    def __str__(self):
        return f"Flatten(input_shape={self.__input_shape})"
            