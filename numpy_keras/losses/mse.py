import numpy as np

class MSE:
    def __init__(
            self, 
            name: str = 'mse',
        ) -> None:

        self.name = name
    
    def __call__(
            self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray,
        ) -> np.float64:
        return np.mean((y_true - y_pred) ** 2)
    
    def grad(
            self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray,
        ) -> np.ndarray:
        return -2 * (y_true - y_pred)