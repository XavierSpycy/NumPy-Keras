try:
    import autograd.numpy as np
except:
    pass

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
        ) -> float:
        
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]