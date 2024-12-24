import numpy as np

class CategoricalCrossEntropy:
    def __init__(
            self, 
            name: str = 'categorical_crossentropy',
        ) -> None:

        self.name = name
    
    def __call__(
            self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray,
        ) -> np.ndarray:
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def grad(
            self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray,
        ) -> np.ndarray:

        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return y_pred_clipped - y_true