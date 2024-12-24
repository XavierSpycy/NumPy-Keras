from typing import (
    Dict, 
    Any,
    Optional
)

try:
    import autograd.numpy as np
except:
    pass

from ..activations._mapper import _ActivationMapper

class Activation:
    """
    Activation layer
    """
    def __init__(
            self, 
            activation: str,
            activation_config: Optional[Dict[str, Any]] = {},
         ) -> None:
        
        """
        Initialize the Activation layer.

        Parameters:
        - activation (str): The activation function.
        - activation_config (dict): The activation function configuration. Default is {}.
        """
        
        self.__activation = activation
        self.__activation_config = activation_config
        
        self.__activation_mapper = _ActivationMapper()
    
    def set_output_dim(
            self, 
            input_dim: int,
        ) -> None:

        """
        Set the output_dim attribute of the layer.

        Parameters:
        - input_dim (int): The input dimension of the layer.
        """

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
        - is_training (bool): Whether the model is training.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """

        self.inputs = inputs
        self.output = self.__activation_mapper[self.__activation](inputs, **self.__activation_config)
        return self.output
    
    @property
    def activation(self):
        return self.__activation
    
    @property
    def output_dim(self):
        return self.__output_dim
    
    def __str__(self):
        return f"Activation(activation={self.__activation})"