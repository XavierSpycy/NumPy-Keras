import numpy as np

class Dropout:
    """
    Dropout layer
    """
    def __init__(
            self, 
            rate: float = 0.5,
        ) -> None:

        """
        Initialize the Dropout layer.

        Parameters:
        - dropout_rate (float): The dropout rate.
        """

        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        
        self.__rate = rate
    
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

        # If the layer is in training mode, compute the outputs using dropout mask
        if is_training:
            # Generate the dropout mask
            self.__mask = (np.random.rand(*inputs.shape) > self.__rate) / (1.0 - self.__rate)
            return inputs * self.__mask # Multiply the inputs by the dropout mask
        # Otherwise, return the inputs
        else:
            return inputs
    
    def backward(
            self, 
            delta: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.

        Returns:
        - delta (np.ndarray): The delta of the previous layer.
        """

        return delta * self.__mask

    def set_output_dim(self, input_dim: int) -> None:

        """
        Set the output_dim attribute of the layer.

        Parameters:
        - input_dim (int): The input dimension of the layer.
        """
        
        self.__output_dim = input_dim
    
    @property
    def output_dim(self) -> int:
        return self.__output_dim
    
    def __str__(self):
        return f"Dropout(rate={self.__rate})"