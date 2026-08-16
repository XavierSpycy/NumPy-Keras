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


class LSTM:
    """
    Long Short-Term Memory layer, in the Keras gate layout [i, f, g, o].

    The four gates share one kernel on each side, so one timestep is two
    matrix products plus elementwise math:

        pre_t  = x_t @ W_xh + h_{t-1} @ W_hh + b        # (N, 4 * units)
        i = sigmoid(pre[:, :U])        # input gate
        f = sigmoid(pre[:, U:2U])      # forget gate
        g = tanh(pre[:, 2U:3U])        # cell candidate
        o = sigmoid(pre[:, 3U:])       # output gate
        c_t = f * c_{t-1} + i * g      # the memory cell
        h_t = o * tanh(c_t)            # the hidden state

    The forget gate decides what old memory to keep, the input gate what new
    information to store -- the mechanism that lets gradients flow through
    time without vanishing as fast as in a plain RNN.

    Like every layer, it chains through its own activation inside
    backward.  Because h_t is NOT the elementwise activation of a single
    pre-activation (it is o * tanh(c_t)), the gate and candidate derivs
    are applied per timestep inside backward and the `activation` property
    reports None.  `activation` configures the cell candidate, and
    `recurrent_activation` the three sigmoid gates.

    Not implemented (teaching scope): initial_state, stateful mode,
    bidirectional, go_backwards, dropout.
    """
    def __init__(
            self,
            units: int,
            activation: Optional[str] = 'tanh',
            activation_config: Optional[Dict[str, Any]] = {},
            recurrent_activation: Optional[str] = 'sigmoid',
            recurrent_activation_config: Optional[Dict[str, Any]] = {},
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
        Initialize the LSTM layer.

        Parameters:
        - units (int): The number of hidden units.
        - activation (str, optional): The activation of the cell candidate g. Default is 'tanh'.
        - activation_config (dict, optional): The activation function configuration. Default is {}.
        - recurrent_activation (str, optional): The activation of the i/f/o gates. Default is 'sigmoid'.
        - recurrent_activation_config (dict, optional): The recurrent activation function configuration. Default is {}.
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
        self.__activation = activation if activation is not None else "tanh"
        self.__activation_config = activation_config
        self.__recurrent_activation = recurrent_activation if recurrent_activation is not None else "sigmoid"
        self.__recurrent_activation_config = recurrent_activation_config
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

        # Gate and candidate activations are applied inside forward and
        # their derivatives inside backward (derivs take post-activation
        # values).  Both must have a derivative in the mapper.
        try:
            self.__gate_deriv = self.__activation_mapper[self.__recurrent_activation + '_deriv']
            self.__cand_deriv = self.__activation_mapper[self.__activation + '_deriv']
        except ValueError:
            raise ValueError(
                f"LSTM activations {self.__activation!r}/{self.__recurrent_activation!r} "
                f"must have derivatives (e.g. 'tanh'/'sigmoid'), not 'softmax'.")

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
                f"LSTM expects a 2D input shape (timesteps, features), "
                f"got {input_shape}. Use Input((T, F)) or a "
                f"return_sequences=True RNN before this layer.")

        self.__input_shape = tuple(input_shape)
        T, F = self.__input_shape
        U = self.__units

        self.__output_shape = (T, U) if self.__return_sequences else (U,)

        self.params = {}
        self.grads = {}
        self.params["W_xh"] = self.__initializer[self.__kernel_initializer](
            **self.__kernel_initializer_config)((F, 4 * U))
        self.grads["W_xh"] = np.zeros_like(self.params["W_xh"])
        self.params["W_hh"] = self.__initializer[self.__recurrent_initializer](
            **self.__recurrent_initializer_config)((U, 4 * U))
        self.grads["W_hh"] = np.zeros_like(self.params["W_hh"])

        if self.__use_bias:
            self.params["b"] = self.__initializer[self.__bias_initializer](
                **self.__bias_initializer_config)((4 * U,))
            self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation: loop over timesteps, updating the memory cell.

        Parameters:
        - inputs (ndarray): The input data of shape (N, T, F).
        - is_training (bool): Whether the model is training or not.
        """

        if self.__input_shape is None:
            raise ValueError("LSTM has no input shape; add an Input layer first.")
        if inputs.ndim != 3 or inputs.shape[2] != self.__input_shape[1]:
            raise ValueError(
                f"LSTM expects inputs of shape (N, T, {self.__input_shape[1]}), "
                f"got {inputs.shape}.")

        N, T, _ = inputs.shape
        U = self.__units

        if is_cupy_array(inputs):
            # GPU path: batch the input projection over all timesteps into a
            # single 3D matmul, so only the recurrence stays in the loop.
            pre_x = inputs @ self.params["W_xh"]           # (N, T, 4U)
            h = np.zeros((N, U))
            c = np.zeros((N, U))
            i_seq = np.empty((N, T, U))
            f_seq = np.empty((N, T, U))
            g_seq = np.empty((N, T, U))
            o_seq = np.empty((N, T, U))
            c_seq = np.empty((N, T, U))
            h_seq = np.empty((N, T, U))
            for t in range(T):
                pre = pre_x[:, t, :] + h @ self.params["W_hh"]
                if "b" in self.params:
                    pre = pre + self.params["b"]
                i = self.__activation_mapper[self.__recurrent_activation](
                    pre[:, :U], **self.__recurrent_activation_config)
                f = self.__activation_mapper[self.__recurrent_activation](
                    pre[:, U:2 * U], **self.__recurrent_activation_config)
                g = self.__activation_mapper[self.__activation](
                    pre[:, 2 * U:3 * U], **self.__activation_config)
                o = self.__activation_mapper[self.__recurrent_activation](
                    pre[:, 3 * U:], **self.__recurrent_activation_config)
                c = f * c + i * g
                h = o * np.tanh(c)
                i_seq[:, t, :] = i
                f_seq[:, t, :] = f
                g_seq[:, t, :] = g
                o_seq[:, t, :] = o
                c_seq[:, t, :] = c
                h_seq[:, t, :] = h
            self.inputs = inputs
            self.__i_seq = i_seq
            self.__f_seq = f_seq
            self.__g_seq = g_seq
            self.__o_seq = o_seq
            self.__c_seq = c_seq
            self.__h_seq = h_seq
            return h_seq if self.__return_sequences else h

        h = np.zeros((N, U))
        c = np.zeros((N, U))
        # gate outputs and cell states per timestep, cached for backward
        i_seq = np.empty((N, T, U))
        f_seq = np.empty((N, T, U))
        g_seq = np.empty((N, T, U))
        o_seq = np.empty((N, T, U))
        c_seq = np.empty((N, T, U))
        h_seq = np.empty((N, T, U))

        for t in range(T):
            pre = inputs[:, t, :] @ self.params["W_xh"] + h @ self.params["W_hh"]
            if "b" in self.params:
                pre = pre + self.params["b"]
            i = self.__activation_mapper[self.__recurrent_activation](
                pre[:, :U], **self.__recurrent_activation_config)
            f = self.__activation_mapper[self.__recurrent_activation](
                pre[:, U:2 * U], **self.__recurrent_activation_config)
            g = self.__activation_mapper[self.__activation](
                pre[:, 2 * U:3 * U], **self.__activation_config)
            o = self.__activation_mapper[self.__recurrent_activation](
                pre[:, 3 * U:], **self.__recurrent_activation_config)

            c = f * c + i * g
            h = o * np.tanh(c)

            i_seq[:, t, :] = i
            f_seq[:, t, :] = f
            g_seq[:, t, :] = g
            o_seq[:, t, :] = o
            c_seq[:, t, :] = c
            h_seq[:, t, :] = h

        self.inputs = inputs
        self.__i_seq = i_seq
        self.__f_seq = f_seq
        self.__g_seq = g_seq
        self.__o_seq = o_seq
        self.__c_seq = c_seq
        self.__h_seq = h_seq

        if self.__return_sequences:
            return h_seq
        return h

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: BPTT through the gates and the memory cell.

        The cell state is the key: the gradient with respect to c_t
        receives the contribution flowing through h_t (via the output
        gate) plus the one from the future, multiplied by the forget
        gate f_{t+1} that gated c_t on the way into c_{t+1}.

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
            d_pre_seq = np.empty((N, T, 4 * U))
            dh = np.zeros((N, U))
            dc = np.zeros((N, U))
            for t in range(T - 1, -1, -1):
                dh = dh + d_out[:, t, :]
                tanh_c = np.tanh(self.__c_seq[:, t, :])
                h_prev = np.zeros((N, U)) if t == 0 else self.__h_seq[:, t - 1, :]
                c_prev = np.zeros((N, U)) if t == 0 else self.__c_seq[:, t - 1, :]
                d_o = dh * tanh_c * self.__gate_deriv(self.__o_seq[:, t, :], **self.__recurrent_activation_config)
                d_c = dh * self.__o_seq[:, t, :] * self.__cand_deriv(tanh_c, **self.__activation_config)
                if t < T - 1:
                    d_c = d_c + dc * self.__f_seq[:, t + 1, :]
                d_i = d_c * self.__g_seq[:, t, :] * self.__gate_deriv(self.__i_seq[:, t, :], **self.__recurrent_activation_config)
                d_f = d_c * c_prev * self.__gate_deriv(self.__f_seq[:, t, :], **self.__recurrent_activation_config)
                d_g = d_c * self.__i_seq[:, t, :] * self.__cand_deriv(self.__g_seq[:, t, :], **self.__activation_config)
                d_pre = np.concatenate([d_i, d_f, d_g, d_o], axis=1)   # (N, 4U)
                d_pre_seq[:, t, :] = d_pre
                self.grads["W_hh"] += h_prev.T @ d_pre
                if "b" in self.grads:
                    self.grads["b"] += d_pre.sum(axis=0)
                dh = d_pre @ self.params["W_hh"].T
                dc = d_c
            # sum_t x_t^T d_pre_t in one reduction, and all dX rows at once
            self.grads["W_xh"] = np.tensordot(self.inputs, d_pre_seq, axes=([0, 1], [0, 1]))
            return d_pre_seq @ self.params["W_xh"].T

        dX = np.empty_like(self.inputs)
        dh = np.zeros((N, U))    # gradient through h_t from the future
        dc = np.zeros((N, U))    # gradient through c_t from the future

        for t in range(T - 1, -1, -1):
            dh = dh + d_out[:, t, :]
            tanh_c = np.tanh(self.__c_seq[:, t, :])
            # h_{t-1} / c_{t-1}; the initial states h_{-1} / c_{-1} are zeros
            h_prev = np.zeros((N, U)) if t == 0 else self.__h_seq[:, t - 1, :]
            c_prev = np.zeros((N, U)) if t == 0 else self.__c_seq[:, t - 1, :]

            # h_t = o_t * tanh(c_t): the output path through the cell
            d_o = dh * tanh_c * self.__gate_deriv(self.__o_seq[:, t, :], **self.__recurrent_activation_config)
            d_c = dh * self.__o_seq[:, t, :] * self.__cand_deriv(tanh_c, **self.__activation_config)
            # ... plus the future: c_{t+1} = f_{t+1} * c_t + ...
            if t < T - 1:
                d_c = d_c + dc * self.__f_seq[:, t + 1, :]

            # c_t = f * c_{t-1} + i * g
            d_i = d_c * self.__g_seq[:, t, :] * self.__gate_deriv(self.__i_seq[:, t, :], **self.__recurrent_activation_config)
            d_f = d_c * c_prev * self.__gate_deriv(self.__f_seq[:, t, :], **self.__recurrent_activation_config)
            d_g = d_c * self.__i_seq[:, t, :] * self.__cand_deriv(self.__g_seq[:, t, :], **self.__activation_config)
            d_pre = np.concatenate([d_i, d_f, d_g, d_o], axis=1)   # (N, 4U)

            self.grads["W_hh"] += h_prev.T @ d_pre
            self.grads["W_xh"] += self.inputs[:, t, :].T @ d_pre
            if "b" in self.grads:
                self.grads["b"] += d_pre.sum(axis=0)
            dX[:, t, :] = d_pre @ self.params["W_xh"].T

            dh = d_pre @ self.params["W_hh"].T
            dc = d_c

        return dX

    @property
    def units(self):
        return self.__units

    @property
    def activation(self):
        # Plain introspection marker: h_t is not a single elementwise
        # activation, so there is no name to report -- the gate and
        # candidate derivs are applied per timestep inside backward.
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
        return f"LSTM(units={self.__units}, return_sequences={self.__return_sequences})"
