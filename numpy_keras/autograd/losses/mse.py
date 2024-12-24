try:
    import autograd.numpy as np
except:
    pass

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
        ) -> float:
        
        return np.mean((y_true - y_pred) ** 2)