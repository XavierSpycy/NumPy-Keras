class Input:
    """
    The Input layer.
    """
    def __init__(
            self, 
            shape: int,
        ) -> None:

        """
        Initialize the Input layer.

        Parameters:
        - shape (int): The shape of the input.
        """

        self.__output_dim = shape
    
    @property
    def output_dim(self):
        return self.__output_dim
    
    def __str__(self):
        return f"Input(shape={self.__output_dim})"