import numpy as np

class CategoricalCrossEntropy:
    """Cross-entropy loss. ``grad`` returns the gradient with respect to
    the network output (typically a softmax layer's output); the softmax
    layer itself then applies its own Jacobian in backward.  Use it with a
    softmax output layer -- on raw logits the clipping makes no sense."""

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
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]

    def grad(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
        ) -> np.ndarray:

        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -y_true / y_pred_clipped / y_true.shape[0]