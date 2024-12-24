from typing import (
    Dict,
    Any, 
    Tuple, 
    Optional,
)

try:
    import autograd.numpy as np
except:
    pass

def one_hot_encode(
        y: np.ndarray, 
        idx2label: Optional[Dict[int, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[int, Any]]:

    """
    One-hot encodes the target labels.

    Parameters:
    - y (np.ndarray): Target labels.

    Returns:
    - y_one_hot (np.ndarray): One-hot encoded target labels.
    - idx2label (dict): Dictionary mapping index to label.
    """

    y_ = y.copy()
    y_ = y.astype(int)

    if idx2label is None:
        unique_classes = np.array(np.unique(y_)).tolist()
        n_classes = len(unique_classes)
        label2idx = {label: idx for idx, label in enumerate(unique_classes)}
        idx2label = {idx: label for idx, label in enumerate(unique_classes)}
    else:
        n_classes = len(idx2label)
        label2idx = {label: idx for idx, label in idx2label.items()}
    y_one_hot = np.zeros((y_.shape[0], n_classes))
    encoded_indices = [label2idx[label] for label in y_]
    y_one_hot[np.arange(y_.shape[0]), encoded_indices] = 1
    return y_one_hot, idx2label

def one_hot_decode(y: np.ndarray, idx2label: dict) -> np.ndarray:
    decoded_indices = [idx2label[np.argmax(one_hot)] for one_hot in y]
    return np.array(decoded_indices)

def train_test_split(
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - X (np.ndarray): Features.
    - y (np.ndarray): Target labels.
    - test_size (float, optional): Fraction of the dataset to include in the test split. Default is 0.2.

    Returns:
    - X_train (np.ndarray): Training features.
    - X_test (np.ndarray): Testing features.
    - y_train (np.ndarray): Training target labels.
    - y_test (np.ndarray): Testing target labels.
    """
    
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    test_size = int(num_samples * test_size)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test