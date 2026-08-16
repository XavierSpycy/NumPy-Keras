from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)

import numpy as _np  # host-only math (output_dim's prod of a shape tuple)
from ..backend import xp as np, is_cupy_array

from ..activations._mapper import _ActivationMapper
from ..initializers._mapper import _InitializerMapper


class SimpleRNN:
    """
    Fully-connected recurrent layer: the simplest RNN.

    At every timestep the same weights combine the current input and the
    previous hidden state,

        h_t = activation(x_t @ W_xh + h_{t-1} @ W_hh + b)

    and the hidden state carries memory across the sequence.  With
    return_sequences=False the layer outputs only the last hidden state
    h_T; the loss gradient then reaches earlier timesteps solely through
    the recurrent path -- backpropagation through time (BPTT).

    Like every layer, it chains through its own activation inside
    backward.  Because h_t is not the elementwise activation of a single
    pre-activation (the hidden state feeds the next timestep directly), the
    activation deriv is applied per timestep inside backward and the
    `activation` property reports None.

    Not implemented (teaching scope): initial_state, stateful mode,
    bidirectional, go_backwards, dropout.
    """
    def __init__(
            self,
            units: int,
            activation: Optional[str] = 'tanh',
            activation_config: Optional[Dict[str, Any]] = {},
            return_sequences: bool = False,
            use_bias: bool = True,
            kernel_initializer: Optional[str] = 'glorot_uniform',
            recurrent_initializer: Optional[str] = 'glorot_uniform',
            bias_initializer: Optional[str] = 'zeros',
            kernel_initializer_config: Optional[Dict[str, Any]] = {},
            recurrent_initializer_config: Optional[Dict[str, Any]] = {},
            bias_initializer_config: Optional[Dict[str, Any]] = {},
        ) -> None:

        """
        Initialize the SimpleRNN layer.

        Parameters:
        - units (int): The number of hidden units.
        - activation (str, optional): The activation function of the hidden
          state. Default is 'tanh'. Must have a derivative (not 'softmax').
        - activation_config (dict, optional): The activation function configuration. Default is {}.
        - return_sequences (bool, optional): Whether to return the full
          sequence of hidden states (N, T, units) or only the last one
          (N, units). Default is False.
        - use_bias (bool, optional): Whether to use bias. Default is True.
        - kernel_initializer (str, optional): The initializer of W_xh. Default is 'glorot_uniform'.
        - recurrent_initializer (str, optional): The initializer of W_hh. Default is 'glorot_uniform'.
        - bias_initializer (str, optional): The bias initializer. Default is 'zeros'.
        - kernel_initializer_config (dict, optional): The kernel initializer configuration. Default is {}.
        - recurrent_initializer_config (dict, optional): The recurrent initializer configuration. Default is {}.
        - bias_initializer_config (dict, optional): The bias initializer configuration. Default is {}.
        """

        self.__units = units
        self.__activation = activation if activation is not None else "linear"
        self.__activation_config = activation_config
        self.__return_sequences = return_sequences
        self.__use_bias = use_bias
        self.__kernel_initializer = kernel_initializer
        self.__kernel_initializer_config = kernel_initializer_config
        self.__recurrent_initializer = recurrent_initializer
        self.__recurrent_initializer_config = recurrent_initializer_config
        self.__bias_initializer = bias_initializer
        self.__bias_initializer_config = bias_initializer_config

        self.__activation_mapper = _ActivationMapper()
        self.__initializer = _InitializerMapper()

        # The recurrent path h_t -> h_{t+1} passes through this layer's own
        # activation, so its derivative must be applied inside backward
        # (the generic chain only covers the output path h_T -> next layer).
        try:
            self.__own_activation_deriv = self.__activation_mapper[self.__activation + '_deriv']
        except ValueError:
            raise ValueError(
                f"SimpleRNN activation {self.__activation!r} has no derivative; "
                f"the hidden recurrence requires one (e.g. 'tanh' or 'relu').")

        self.__input_shape = None   # (T, F), set when the model is built
        self.__output_shape = None

    def set_input_shape(
            self,
            input_shape: Tuple[int, int],
        ) -> None:

        """
        Set the input shape (T, F) and initialize the parameters.
        Called by Sequential during construction.

        Parameters:
        - input_shape (tuple): The sequence shape (timesteps, features)
          without the batch axis.
        """

        if len(input_shape) != 2:
            raise ValueError(
                f"SimpleRNN expects a 2D input shape (timesteps, features), "
                f"got {input_shape}. Use Input((T, F)) or a "
                f"return_sequences=True RNN before this layer.")

        self.__input_shape = tuple(input_shape)
        T, F = self.__input_shape
        U = self.__units

        self.__output_shape = (T, U) if self.__return_sequences else (U,)

        self.params = {}
        self.grads = {}
        self.params["W_xh"] = self.__initializer[self.__kernel_initializer](
            **self.__kernel_initializer_config)((F, U))
        self.grads["W_xh"] = np.zeros_like(self.params["W_xh"])
        self.params["W_hh"] = self.__initializer[self.__recurrent_initializer](
            **self.__recurrent_initializer_config)((U, U))
        self.grads["W_hh"] = np.zeros_like(self.params["W_hh"])

        if self.__use_bias:
            self.params["b"] = self.__initializer[self.__bias_initializer](
                **self.__bias_initializer_config)((U,))
            self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation: loop over timesteps, reusing the same weights.

        Parameters:
        - inputs (ndarray): The input data of shape (N, T, F).
        - is_training (bool): Whether the model is training or not.
        """

        if self.__input_shape is None:
            raise ValueError("SimpleRNN has no input shape; add an Input layer first.")
        if inputs.ndim != 3 or inputs.shape[2] != self.__input_shape[1]:
            raise ValueError(
                f"SimpleRNN expects inputs of shape (N, T, {self.__input_shape[1]}), "
                f"got {inputs.shape}.")

        N, T, _ = inputs.shape
        U = self.__units

        if is_cupy_array(inputs):
            # GPU path: batch the input projection over all timesteps into a
            # single 3D matmul, so only the recurrence stays in the loop.
            pre_x = inputs @ self.params["W_xh"]           # (N, T, U)
            h = np.zeros((N, U))
            h_seq = np.empty((N, T, U))
            for t in range(T):
                pre = pre_x[:, t, :] + h @ self.params["W_hh"]
                if "b" in self.params:
                    pre = pre + self.params["b"]
                h = self.__activation_mapper[self.__activation](pre, **self.__activation_config)
                h_seq[:, t, :] = h
            self.inputs = inputs
            self.__h_seq = h_seq
            return h_seq if self.__return_sequences else h

        h = np.zeros((N, U))
        h_seq = np.empty((N, T, U))   # hidden states, cached for backward
        for t in range(T):
            pre = inputs[:, t, :] @ self.params["W_xh"] + h @ self.params["W_hh"]
            if "b" in self.params:
                pre = pre + self.params["b"]
            h = self.__activation_mapper[self.__activation](pre, **self.__activation_config)
            h_seq[:, t, :] = h

        self.inputs = inputs
        self.__h_seq = h_seq

        if self.__return_sequences:
            return h_seq
        return h

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: backpropagation through time (BPTT).

        The output gradient is spread over the timesteps, then the chain
        rule is unrolled backwards through the recurrence.  With
        return_sequences=False the gradient arrives only at h_T and flows
        to earlier timesteps solely through the recurrent path.

        Parameters:
        - grad (ndarray): The gradient of the loss w.r.t. this layer's
          output, of shape (N, T, units) with return_sequences=True or
          (N, units) otherwise.
        """

        N, T, F = self.inputs.shape
        U = self.__units

        d_out = np.zeros((N, T, U))
        if self.__return_sequences:
            d_out = grad
        else:
            d_out[:, -1, :] = grad

        self.grads["W_xh"] = np.zeros_like(self.params["W_xh"])
        self.grads["W_hh"] = np.zeros_like(self.params["W_hh"])
        if "b" in self.grads:
            self.grads["b"] = np.zeros_like(self.params["b"])

        if is_cupy_array(grad):
            # GPU path: BPTT loops over the recurrence only; the W_xh
            # accumulation and dX are batched into single ops afterwards.
            d_pre_seq = np.empty((N, T, U))
            dh = np.zeros((N, U))
            for t in range(T - 1, -1, -1):
                dh = dh + d_out[:, t, :]
                d_pre = dh * self.__own_activation_deriv(self.__h_seq[:, t, :], **self.__activation_config)
                h_prev = np.zeros((N, U)) if t == 0 else self.__h_seq[:, t - 1, :]
                self.grads["W_hh"] += h_prev.T @ d_pre
                if "b" in self.grads:
                    self.grads["b"] += d_pre.sum(axis=0)
                d_pre_seq[:, t, :] = d_pre
                dh = d_pre @ self.params["W_hh"].T
            # sum_t x_t^T d_pre_t in one reduction, and all dX rows at once
            self.grads["W_xh"] = np.tensordot(self.inputs, d_pre_seq, axes=([0, 1], [0, 1]))
            return d_pre_seq @ self.params["W_xh"].T

        dX = np.empty_like(self.inputs)
        dh = np.zeros((N, U))          # gradient through h_t from the future
        for t in range(T - 1, -1, -1):
            dh = dh + d_out[:, t, :]
            # through the activation: derivs take the post-activation value
            d_pre = dh * self.__own_activation_deriv(self.__h_seq[:, t, :], **self.__activation_config)
            # h_{t-1}; the initial state h_{-1} is zeros
            h_prev = np.zeros((N, U)) if t == 0 else self.__h_seq[:, t - 1, :]
            self.grads["W_hh"] += h_prev.T @ d_pre
            self.grads["W_xh"] += self.inputs[:, t, :].T @ d_pre
            if "b" in self.grads:
                self.grads["b"] += d_pre.sum(axis=0)
            dX[:, t, :] = d_pre @ self.params["W_xh"].T
            dh = d_pre @ self.params["W_hh"].T

        return dX

    @property
    def units(self):
        return self.__units

    @property
    def activation(self):
        # Plain introspection marker: h_t is not a single elementwise
        # activation, so there is no name to report -- the activation
        # deriv is applied per timestep inside backward.
        return None

    @property
    def activation_config(self):
        return self.__activation_config

    @property
    def return_sequences(self):
        return self.__return_sequences

    @property
    def output_dim(self):
        return int(_np.prod(self.__output_shape))

    @property
    def output_shape(self):
        return self.__output_shape

    def __str__(self):
        return f"SimpleRNN(units={self.__units}, activation={self.__activation}, return_sequences={self.__return_sequences})"
