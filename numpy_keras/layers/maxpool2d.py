from typing import (
    Optional,
    Tuple,
    Union,
)

import numpy as _np  # host-only math (e.g. output_dim's prod of a shape tuple)
from ..backend import (
    xp as np,
    is_numpy_array,
    scatter_add,
    sliding_window_view,
)

from ..cython import _kernels as _ck
from .conv2d import _as_pair


class MaxPool2D:
    """
    Max pooling layer for 2D inputs.

    Each output pixel takes the maximum of its pooling window.  The backward
    pass routes each gradient to the input position that won the maximum,
    accumulating with np.add.at because overlapping windows share pixels.
    """
    def __init__(
            self,
            pool_size: Union[int, Tuple[int, int]] = 2,
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Union[int, str] = 0,
        ) -> None:

        """
        Initialize the MaxPool2D layer.

        Parameters:
        - pool_size (int or tuple): Height and width of the pooling window. Default is 2.
        - stride (int or tuple, optional): Stride of the pooling. Defaults to pool_size.
        - padding (int or str, optional): 'same' or the number of zero-padding
          pixels. Default is 0.
        """

        self.__pool_size = _as_pair(pool_size)
        self.__stride = _as_pair(stride) if stride is not None else self.__pool_size
        self.__padding = padding

        self.__input_shape = None   # (H, W, C), set when the model is built
        self.__output_shape = None
        self.__pad = None           # ((top, bottom), (left, right))

    def set_input_shape(
            self,
            input_shape: Tuple[int, int, int],
        ) -> None:

        """
        Set the input shape (H, W, C) and derive the output shape.
        Called by Sequential during construction.

        Parameters:
        - input_shape (tuple): The input shape (H, W, C) without the batch axis.
        """

        if len(input_shape) != 3:
            raise ValueError(f"MaxPool2D expects a 3D input shape (H, W, C), got {input_shape}.")

        self.__input_shape = tuple(input_shape)
        H, W, _ = self.__input_shape
        ph, pw = self.__pool_size
        sh, sw = self.__stride

        if self.__padding == 'same':
            OH = -(-H // sh)
            OW = -(-W // sw)
            pad_h = max((OH - 1) * sh + ph - H, 0)
            pad_w = max((OW - 1) * sw + pw - W, 0)
            self.__pad = ((pad_h // 2, pad_h - pad_h // 2),
                          (pad_w // 2, pad_w - pad_w // 2))
        else:
            self.__pad = ((self.__padding, self.__padding),
                          (self.__padding, self.__padding))
            OH = (H + 2 * self.__padding - ph) // sh + 1
            OW = (W + 2 * self.__padding - pw) // sw + 1

        if OH <= 0 or OW <= 0:
            raise ValueError(
                f"MaxPool2D pool {self.__pool_size} with padding {self.__padding} "
                f"is too large for input {self.__input_shape}.")

        self.__output_shape = (OH, OW, self.__input_shape[2])

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation: pad, slide the pooling window, take the max.

        Parameters:
        - inputs (ndarray): The input data of shape (N, H, W, C).
        - is_training (bool): Whether the model is training or not.
        """

        if self.__input_shape is None:
            raise ValueError("MaxPool2D has no input shape; add an Input layer first.")
        if inputs.ndim != 4:
            raise ValueError(f"MaxPool2D expects 4D inputs (N, H, W, C), got {inputs.shape}.")

        (ph_b, ph_a), (pw_b, pw_a) = self.__pad
        x_pad = np.pad(inputs, ((0, 0), (ph_b, ph_a), (pw_b, pw_a), (0, 0)))
        ph, pw = self.__pool_size
        sh, sw = self.__stride

        # Optional Cython fast path: window max + argmax in C
        if (_ck is not None and hasattr(_ck, 'maxpool_forward')
                and is_numpy_array(inputs)
                and inputs.dtype == np.float64):
            output, self.__amax = _ck.maxpool_forward(x_pad, ph, pw, sh, sw)
            self.inputs_padded = x_pad
            return output

        # sliding_window_view appends the window axes at the end; transpose
        # them next to the spatial axes so the windows live in (3, 4)
        win = sliding_window_view(x_pad, (ph, pw), axis=(1, 2))[:, ::sh, ::sw]
        win = win.transpose(0, 1, 2, 4, 5, 3)   # (N, OH, OW, ph, pw, C)
        N, OH, OW, _, _, C = win.shape
        output = np.max(win, axis=(3, 4))

        # remember which position won in each window for the backward pass
        self.__amax = np.argmax(win.reshape(N, OH, OW, ph * pw, C), axis=3)
        self.inputs_padded = x_pad
        return output

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: scatter each output gradient onto the input
        position that won its window's maximum.

        Parameters:
        - grad (ndarray): The gradient of the loss w.r.t. this layer's
          output, of shape (N, OH, OW, C).
        """

        ph, pw = self.__pool_size
        sh, sw = self.__stride
        N, OH, OW, C = grad.shape
        Hp, Wp = self.inputs_padded.shape[1], self.inputs_padded.shape[2]

        # Optional Cython fast path: scatter in C
        if (_ck is not None and hasattr(_ck, 'maxpool_backward')
                and is_numpy_array(grad)
                and grad.dtype == np.float64):
            grad_padded = np.zeros((N, Hp, Wp, C))
            _ck.maxpool_backward(grad, self.__amax, grad_padded, ph, pw, sh, sw)
        else:
            grad_padded = np.zeros((N, Hp, Wp, C))
            n_idx, i_idx, j_idx, c_idx = np.meshgrid(
                np.arange(N), np.arange(OH), np.arange(OW), np.arange(C), indexing='ij')
            h_abs = i_idx * sh + self.__amax // pw
            w_abs = j_idx * sw + self.__amax % pw
            scatter_add(
                grad_padded,
                (n_idx.ravel(), h_abs.ravel(), w_abs.ravel(), c_idx.ravel()),
                grad.ravel())

        (ph_b, ph_a), (pw_b, pw_a) = self.__pad
        return grad_padded[:, ph_b:Hp - ph_a, pw_b:Wp - pw_a, :]

    @property
    def pool_size(self):
        return self.__pool_size

    @property
    def stride(self):
        return self.__stride

    @property
    def padding(self):
        return self.__padding

    @property
    def output_dim(self):
        return int(_np.prod(self.__output_shape))

    @property
    def output_shape(self):
        return self.__output_shape

    def __str__(self):
        return f"MaxPool2D(pool_size={self.__pool_size}, stride={self.__stride})"
