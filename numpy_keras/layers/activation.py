from typing import (
    Dict, 
    Any,
    Optional
)

import numpy as np

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

        self.__activation_deriv = None
        self.__activation_derive_config = {}
        self.__activation_mapper = _ActivationMapper()

    def set_activation_deriv(
            self, 
            prev_layer_activation: str,
            prev_layer_activation_config: Dict[str, Any]
        ) -> None:
        
        """
        Set the activation derivative function of the previous layer.

        Parameters:
        - prev_layer_activation (str): The activation function of the previous layer.
        - prev_layer_activation_config (dict): The activation function configuration of the previous layer.
        """

        self.__activation_deriv = self.__activation_mapper[prev_layer_activation + '_deriv'] if prev_layer_activation else None
        self.__activation_derive_config = prev_layer_activation_config
    
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
            inputs,
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
    
    def backward(self, grad):
        return grad * self.__activation_deriv(self.inputs, **self.__activation_derive_config)
    
    @property
    def activation(self):
        return self.__activation
    
    @property
    def output_dim(self):
        return self.__output_dim
    
    def __str__(self):
        return f"Activation(activation={self.__activation})"