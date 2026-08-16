from typing import (
    Dict,
    Any,
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

from ..activations._mapper import _ActivationMapper
from ..initializers._mapper import _InitializerMapper
from ..cython import _kernels as _ck

# Kernel dispatch codes for the optional Cython fast path; unknown
# activations fall through to the pure NumPy code below.
_ACT_CODES = {"linear": 0, "relu": 1, "sigmoid": 2, "tanh": 3}
_DERIV_CODES = {"linear_deriv": 0, "relu_deriv": 1, "sigmoid_deriv": 2, "tanh_deriv": 3}


def _as_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize an int or a 2-tuple into a (height, width) pair."""
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or a pair, got {value!r}.")


class Conv2D:
    """
    Convolutional layer for 2D inputs, implemented with the im2col trick.

    im2col rearranges every kh x kw x C receptive field of the input into a
    row of a matrix, so the whole convolution becomes a single matrix
    product: conv(X, W) = cols(X) @ W_flat.  This is the classic teaching
    implementation (CS231n style): short to write, easy to follow, and the
    heavy lifting is delegated to BLAS.
    """
    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, str] = 0,
            activation: Optional[str] = None,
            activation_config: Optional[Dict[str, Any]] = None,
            use_bias: bool = True,
            kernel_initializer: Optional[str] = 'he_normal',
            bias_initializer: Optional[str] = 'zeros',
            kernel_initializer_config: Optional[Dict[str, Any]] = None,
            bias_initializer_config: Optional[Dict[str, Any]] = None,
        ) -> None:

        """
        Initialize the Conv2D layer.

        Parameters:
        - filters (int): The number of filters (output channels).
        - kernel_size (int or tuple): Height and width of the kernel.
        - stride (int or tuple, optional): Stride of the convolution. Default is 1.
        - padding (int or str, optional): 'same' pads so that the output has
          the same spatial size as the input (ceil(H / stride)); an int pads
          both sides with that many zeros. Default is 0.
        - activation (str, optional): The activation function. Default is None (linear).
        - activation_config (dict, optional): The activation function configuration. Default is None.
        - use_bias (bool, optional): Whether to use bias. Default is True.
        - kernel_initializer (str, optional): The kernel initializer. Default is 'he_normal'.
        - bias_initializer (str, optional): The bias initializer. Default is 'zeros'.
        - kernel_initializer_config (dict, optional): The kernel initializer configuration. Default is None.
        - bias_initializer_config (dict, optional): The bias initializer configuration. Default is None.
        """

        self.__filters = filters
        self.__kernel_size = _as_pair(kernel_size)
        self.__stride = _as_pair(stride)
        self.__padding = padding
        self.__activation = activation if activation is not None else "linear"
        self.__activation_config = activation_config or {}
        self.__use_bias = use_bias
        self.__kernel_initializer = kernel_initializer
        self.__kernel_initializer_config = kernel_initializer_config or {}
        self.__bias_initializer = bias_initializer
        self.__bias_initializer_config = bias_initializer_config or {}

        # Deriv code for the optional Cython fast path: the layer chains
        # through ITS OWN activation (softmax has no elementwise deriv,
        # so it always takes the pure path).
        self.__activation_deriv_code = (
            _DERIV_CODES.get((self.__activation or "") + "_deriv", -2)
            if self.__activation not in (None, "linear") else -1)
        self.__activation_mapper = _ActivationMapper()
        self.__initializer = _InitializerMapper()

        self.__input_shape = None   # (H, W, C), set when the model is built
        self.__output_shape = None
        self.__pad = None           # ((top, bottom), (left, right))

    def set_input_shape(
            self,
            input_shape: Tuple[int, int, int],
        ) -> None:

        """
        Set the input shape (H, W, C) and initialize the parameters.
        Called by Sequential during construction.

        Parameters:
        - input_shape (tuple): The input shape (H, W, C) without the batch axis.
        """

        if len(input_shape) != 3:
            raise ValueError(f"Conv2D expects a 3D input shape (H, W, C), got {input_shape}.")

        self.__input_shape = tuple(input_shape)
        H, W, _ = self.__input_shape
        kh, kw = self.__kernel_size
        sh, sw = self.__stride

        if self.__padding == 'same':
            # pad so that output_size = ceil(input_size / stride); the extra
            # pixel of an odd total padding goes to the bottom / right
            OH = -(-H // sh)
            OW = -(-W // sw)
            pad_h = max((OH - 1) * sh + kh - H, 0)
            pad_w = max((OW - 1) * sw + kw - W, 0)
            self.__pad = ((pad_h // 2, pad_h - pad_h // 2),
                          (pad_w // 2, pad_w - pad_w // 2))
        else:
            self.__pad = ((self.__padding, self.__padding),
                          (self.__padding, self.__padding))
            OH = (H + 2 * self.__padding - kh) // sh + 1
            OW = (W + 2 * self.__padding - kw) // sw + 1

        if OH <= 0 or OW <= 0:
            raise ValueError(
                f"Conv2D kernel {self.__kernel_size} with padding {self.__padding} "
                f"is too large for input {self.__input_shape}.")

        self.__output_shape = (OH, OW, self.__filters)

        self.params = {}
        self.grads = {}
        C = self.__input_shape[2]
        self.params["W"] = self.__initializer[self.__kernel_initializer](
            **self.__kernel_initializer_config)((kh, kw, C, self.__filters))
        self.grads["W"] = np.zeros_like(self.params["W"])

        if self.__use_bias:
            self.params["b"] = self.__initializer[self.__bias_initializer](
                **self.__bias_initializer_config)((self.__filters,))
            self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation: pad, im2col, matrix product, activation.

        Parameters:
        - inputs (ndarray): The input data of shape (N, H, W, C).
        - is_training (bool): Whether the model is training or not.
        """

        if self.__input_shape is None:
            raise ValueError("Conv2D has no input shape; add an Input layer first.")
        if inputs.ndim != 4 or inputs.shape[3] != self.__input_shape[2]:
            raise ValueError(
                f"Conv2D expects inputs of shape (N, H, W, {self.__input_shape[2]}), "
                f"got {inputs.shape}.")

        (ph_b, ph_a), (pw_b, pw_a) = self.__pad
        x_pad = np.pad(inputs, ((0, 0), (ph_b, ph_a), (pw_b, pw_a), (0, 0)))
        kh, kw = self.__kernel_size
        sh, sw = self.__stride

        # im2col: one sliding window per output position, flattened into a
        # matrix of shape (N * OH * OW, kh * kw * C).  sliding_window_view
        # appends the window axes at the end, so transpose them next to the
        # spatial axes first: the flattened column order must match
        # W.reshape(kh * kw * C, filters).
        cols = sliding_window_view(x_pad, (kh, kw), axis=(1, 2))[:, ::sh, ::sw]
        cols = cols.transpose(0, 1, 2, 4, 5, 3)   # (N, OH, OW, kh, kw, C)
        N, OH, OW, _, _, C = cols.shape
        cols = cols.reshape(N * OH * OW, kh * kw * C)

        # Optional Cython fast path: matmul + activation fused in one kernel
        # (the column matrix is 2D -- the exact input type of dense_forward).
        if (_ck is not None
                and is_numpy_array(inputs)
                and is_numpy_array(self.params["W"])
                and inputs.dtype == np.float64
                and self.params["W"].dtype == np.float64
                and not self.__activation_config
                and self.activation in _ACT_CODES):
            lin = _ck.dense_forward(
                cols, self.params["W"].reshape(kh * kw * C, self.__filters),
                self.params.get("b"), _ACT_CODES[self.activation])
            self.inputs = inputs
            self.inputs_padded = x_pad
            self.cols = cols
            self.output = lin.reshape(N, OH, OW, self.__filters)
            return self.output

        lin_output = cols @ self.params["W"].reshape(kh * kw * C, self.__filters)
        if "b" in self.params:
            lin_output = lin_output + self.params["b"]

        self.inputs = inputs
        self.inputs_padded = x_pad
        self.cols = cols

        output = self.__activation_mapper[self.activation](lin_output, **self.__activation_config)
        self.output = output.reshape(N, OH, OW, self.__filters)
        return self.output

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: BLAS for dW / db, col2im scatter-add for dX.

        Parameters:
        - grad (ndarray): The gradient of the loss w.r.t. this layer's
          output, of shape (N, OH, OW, filters).
        """

        kh, kw = self.__kernel_size
        sh, sw = self.__stride
        N, OH, OW, F = grad.shape
        C = self.__input_shape[2]
        grad_cols = grad.reshape(N * OH * OW, F)

        W_col = self.params["W"].reshape(kh * kw * C, F)

        # Optional Cython fast path: BLAS via dense_backward on the column
        # matrices, col2im scatter in C.  dense_backward evaluates the
        # activation derivative on the post-activation output, exactly like
        # the pure path below.
        if (_ck is not None
                and hasattr(_ck, 'col2im')
                and is_numpy_array(grad)
                and is_numpy_array(self.inputs)
                and grad.dtype == np.float64
                and self.inputs.dtype == np.float64
                and self.params["W"].dtype == np.float64
                and not self.__activation_config
                and self.__activation_deriv_code != -2):
            dW, db, grad_cols_next = _ck.dense_backward(
                self.cols, self.output.reshape(-1, F), grad_cols, W_col,
                self.__activation_deriv_code)
            self.grads["W"] = dW.reshape(kh, kw, C, F)
            if "b" in self.grads:
                self.grads["b"] = db
            Hp, Wp = self.inputs_padded.shape[1], self.inputs_padded.shape[2]
            grad_padded = np.zeros((N, Hp, Wp, C))
            _ck.col2im(grad_cols_next, grad_padded, kh, kw, sh, sw)
            (ph_b, ph_a), (pw_b, pw_a) = self.__pad
            return grad_padded[:, ph_b:Hp - ph_a, pw_b:Wp - pw_a, :]

        # own activation, evaluated on the cached post-activation output:
        # dz = grad ⊙ f'(y); the parameter gradients use dz, and dX comes
        # from the column matmul + col2im on dz
        grad_cols = self.__activation_mapper.backward(
            self.activation, self.output.reshape(-1, F), grad_cols,
            self.__activation_config)
        self.grads["W"] = (self.cols.T @ grad_cols).reshape(kh, kw, C, F)
        if "b" in self.grads:
            self.grads["b"] = grad_cols.sum(axis=0)

        grad_next_cols = grad_cols @ W_col.T  # (N * OH * OW, kh * kw * C)
        grad_next = self.__col2im(grad_next_cols.reshape(N, OH, OW, kh, kw, C))
        return grad_next

    def __col2im(
            self,
            grad_cols: np.ndarray,
        ) -> np.ndarray:

        """
        Scatter-add the column-space gradient back onto the input image.

        Each pixel appears in several receptive fields, so contributions from
        all the windows that contain it must accumulate (np.add.at).

        Parameters:
        - grad_cols (ndarray): The gradient in column space, of shape
          (N, OH, OW, kh, kw, C).
        """

        N, OH, OW, kh, kw, C = grad_cols.shape
        sh, sw = self.__stride
        Hp, Wp = self.inputs_padded.shape[1], self.inputs_padded.shape[2]

        grad_padded = np.zeros((N, Hp, Wp, C))
        M = OH * OW
        n_idx = np.repeat(np.arange(N), M)
        i_idx = np.tile(np.repeat(np.arange(OH), OW), N)
        j_idx = np.tile(np.tile(np.arange(OW), OH), N)
        grad_cols_flat = grad_cols.reshape(N * M, kh * kw, C)
        for i in range(kh):
            for j in range(kw):
                scatter_add(
                    grad_padded,
                    (n_idx, i_idx * sh + i, j_idx * sw + j, slice(None)),
                    grad_cols_flat[:, i * kw + j, :])

        (ph_b, ph_a), (pw_b, pw_a) = self.__pad
        return grad_padded[:, ph_b:Hp - ph_a, pw_b:Wp - pw_a, :]

    @property
    def filters(self):
        return self.__filters

    @property
    def kernel_size(self):
        return self.__kernel_size

    @property
    def stride(self):
        return self.__stride

    @property
    def padding(self):
        return self.__padding

    @property
    def activation(self):
        return self.__activation

    @property
    def activation_config(self):
        return self.__activation_config

    @property
    def output_dim(self):
        return int(_np.prod(self.__output_shape))

    @property
    def output_shape(self):
        return self.__output_shape

    def __str__(self):
        return f"Conv2D(filters={self.__filters}, kernel_size={self.__kernel_size}, activation={self.__activation})"
