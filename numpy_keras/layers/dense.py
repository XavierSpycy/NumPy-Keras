from typing import (
    Dict, 
    Any,
    Optional
)

import numpy as np

from ..activations._mapper import _ActivationMapper
from ..initializers._mapper import _InitializerMapper

class Dense:
    """
    Dense layer
    """
    def __init__(
            self, 
            units: int, 
            activation: Optional[str] = 'tanh',
            activation_config: Optional[Dict[str, Any]] = {},
            use_bias: bool = True,
            kernel_initializer: Optional[str] = 'glorot_uniform',
            bias_initializer: Optional[str] = 'zeros',
            kernel_initializer_config: Optional[Dict[str, Any]] = {},
            bias_initializer_config: Optional[Dict[str, Any]] = {},
        ) -> None:
        
        """
        Initialize the Dense layer.

        Parameters:
        - units (int): The number of units.
        - activation (str): The activation function. Default is 'tanh'.
        - activation_config (dict): The activation function configuration. Default is {}.
        - use_bias (bool): Whether to use bias. Default is True.
        - kernel_initializer (str): The kernel initializer. Default is 'glorot_uniform'.
        - bias_initializer (str): The bias initializer. Default is 'zeros'.
        - kernel_initializer_config (dict): The kernel initializer configuration. Default is {}.
        - bias_initializer_config (dict): The bias initializer configuration. Default is {}.
        """

        self.__units = units
        self.__activation = activation if activation is not None else "linear"
        self.__activation_config = activation_config
        self.__use_bias = use_bias
        self.__kernel_initializer = kernel_initializer
        self.__kernel_initializer_config = kernel_initializer_config
        self.__bias_initializer = bias_initializer
        self.__bias_initializer_config = bias_initializer_config
        
        self.__activation_deriv = None
        self.__activation_derive_config = {}
        self.__activation_mapper = _ActivationMapper()
        self.__initializer = _InitializerMapper()

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
        
        self.params["W"] = self.__initializer[self.__kernel_initializer](**self.__kernel_initializer_config)((input_dim, self.__units))
        self.grads["W"] = np.zeros_like(self.params["W"])

        if self.__use_bias:
            self.params["b"] = self.__initializer[self.__bias_initializer](**self.__bias_initializer_config)((self.__units,))
            self.grads["b"] = np.zeros_like(self.params["b"])
        
    def forward(
            self, 
            inputs: np.ndarray, 
            is_training: bool,
        ) -> np.ndarray:
        
        """
        Forward propagation.

        Parameters:
        - inputs (ndarray): The input data.
        - is_training (bool): Whether the model is training or not.
        """

        lin_output = np.dot(inputs, self.params["W"]) + self.params["b"] if "b" in self.params else np.dot(inputs, self.params["W"])
        self.output = self.__activation_mapper[self.activation](lin_output, **self.__activation_config) if self.activation is not None else lin_output
        self.inputs = inputs
        return self.output
    
    def backward(
            self, 
            grad: np.ndarray,
        ) -> np.ndarray:
        
        """
        Backward propagation.

        Parameters:
        - grad (ndarray): The gradient of the loss.
        """
        
        self.grads["W"] = np.dot(self.inputs.T, grad)
        if "b" in self.grads:
            self.grads["b"] = np.sum(grad, axis=0)
        grad = np.dot(grad, self.params["W"].T)
        if self.__activation_deriv:
            grad *= self.__activation_deriv(self.inputs, **self.__activation_derive_config)
        return grad
    
    @property
    def units(self):
        return self.__units
    
    @property
    def activation(self):
        return self.__activation
    
    @property
    def activation_config(self):
        return self.__activation_config
    
    @property
    def activation_deriv(self):
        return self.__activation_deriv
    
    @property
    def output_dim(self):
        return self.__units
    
    def __str__(self):
        return f"Dense(units={self.__units}, activation={self.__activation})"