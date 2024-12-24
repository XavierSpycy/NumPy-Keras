import re
from typing import (
    Dict,
    Any, 
    Tuple, 
    Optional,
    Generator,
)

import numpy as np

from .. import callbacks

def camel_to_snake(layer_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', layer_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def conditional_tqdm(
          iterable: range, 
          use_progress_bar: bool = False,
        ) -> Generator[int, None, None]:
        """
        Determine whether to use tqdm or not based on the use_progress_bar flag.

        Parameters:
        - iterable (range): Range of values to iterate over.
        - use_progress_bar (bool, optional): Whether to print progress bar. Default is False.

        Returns:
        - item (int): Current iteration.
        """
        if use_progress_bar:
            try:
                from tqdm import tqdm
                for item in tqdm(iterable):
                    yield item
            except ImportError:
                for item in iterable:
                    yield item
        else:
            for item in iterable:
                yield item

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

def plot_history(history: callbacks.History) -> None:

    """
    Plots the training history including loss and metrics.
    
    Args:
        history (callbacks.History): The history object returned by model.fit().
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    num_plots = len(history.metrics) + 1
    rows = (num_plots + 1) // 2
    _, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    cmap = cm.get_cmap('tab20', num_plots)
    colors = [cmap(i) for i in range(num_plots)]

    epochs = list(range(len(history['loss'])))
    tick_spacing = max(1, len(epochs) // 10)

    axes[0].plot(history['loss'], color=colors[0])
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    axes[0].set_xticks(min(epochs[::tick_spacing], history.validation_epochs, key=len))

    for idx, metric in enumerate(history.metrics):
        title = " ".join(map(lambda x: x.capitalize(), metric.split('_')))
        axes[idx + 1].plot(history.metrics[metric], color=colors[idx + 1])
        axes[idx + 1].set_title(title)
        axes[idx + 1].set_xlabel('Epoch')
        axes[idx + 1].set_ylabel(title)
        axes[idx + 1].grid()
        axes[idx + 1].set_xticks(min(epochs[::tick_spacing], history.validation_epochs, key=len))

    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()