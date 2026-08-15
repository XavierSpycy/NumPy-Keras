from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)

import numpy as np

from ..activations._mapper import _ActivationMapper
from ..initializers._mapper import _InitializerMapper


class GRU:
    """
    Gated Recurrent Unit, in the Keras gate layout [z, r, h̃], with the
    classic (reset_after=False) formulation used in most textbooks:

        z   = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)   # update gate
        r   = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)   # reset gate
        h̃   = tanh(x_t @ W_xh + (r * h_{t-1}) @ W_hh + b_h)  # candidate
        h_t = (1 - z) * h_{t-1} + z * h̃

    The update gate z interpolates between the old state and the candidate:
    z close to 1 takes the new candidate, z close to 0 copies h_{t-1}
    through unchanged.  The reset gate r decides how much of the old state
    the candidate may see.  This is the LSTM cell's essential mechanism
    (adaptive forgetting + gated update) with one less gate.

    Like LSTM, h_t is not the elementwise activation of a single
    pre-activation, so the `activation` property returns None and the whole
    output chain is handled inside `backward`.  `activation` configures the
    candidate, `recurrent_activation` the z/r gates.

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
        Initialize the GRU layer.

        Parameters:
        - units (int): The number of hidden units.
        - activation (str, optional): The activation of the candidate h̃. Default is 'tanh'.
        - activation_config (dict, optional): The activation function configuration. Default is {}.
        - recurrent_activation (str, optional): The activation of the z/r gates. Default is 'sigmoid'.
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

        self.__activation_deriv = None
        self.__activation_derive_config = {}
        self.__activation_mapper = _ActivationMapper()
        self.__initializer = _InitializerMapper()

        try:
            self.__gate_deriv = self.__activation_mapper[self.__recurrent_activation + '_deriv']
            self.__cand_deriv = self.__activation_mapper[self.__activation + '_deriv']
        except ValueError:
            raise ValueError(
                f"GRU activations {self.__activation!r}/{self.__recurrent_activation!r} "
                f"must have derivatives (e.g. 'tanh'/'sigmoid'), not 'softmax'.")

        self.__input_shape = None   # (T, F), set when the model is built
        self.__output_shape = None

    def set_activation_deriv(
            self,
            prev_layer_activation: str,
            prev_layer_activation_config: Dict[str, Any],
        ) -> None:

        """
        Set the activation derivative function of the previous layer.

        Parameters:
        - prev_layer_activation (str): The activation function of the previous layer.
        - prev_layer_activation_config (dict): The activation function configuration of the previous layer.
        """

        self.__activation_deriv = self.__activation_mapper[prev_layer_activation + '_deriv'] if prev_layer_activation else None
        self.__activation_derive_config = prev_layer_activation_config

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
                f"GRU expects a 2D input shape (timesteps, features), "
                f"got {input_shape}. Use Input((T, F)) or a "
                f"return_sequences=True RNN before this layer.")

        self.__input_shape = tuple(input_shape)
        T, F = self.__input_shape
        U = self.__units

        self.__output_shape = (T, U) if self.__return_sequences else (U,)

        self.params = {}
        self.grads = {}
        self.params["W_xh"] = self.__initializer[self.__kernel_initializer](
            **self.__kernel_initializer_config)((F, 3 * U))
        self.grads["W_xh"] = np.zeros_like(self.params["W_xh"])
        self.params["W_hh"] = self.__initializer[self.__recurrent_initializer](
            **self.__recurrent_initializer_config)((U, 3 * U))
        self.grads["W_hh"] = np.zeros_like(self.params["W_hh"])

        if self.__use_bias:
            self.params["b"] = self.__initializer[self.__bias_initializer](
                **self.__bias_initializer_config)((3 * U,))
            self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(
            self,
            inputs: np.ndarray,
            is_training: bool,
        ) -> np.ndarray:

        """
        Forward propagation: loop over timesteps, gating the hidden state.

        Parameters:
        - inputs (ndarray): The input data of shape (N, T, F).
        - is_training (bool): Whether the model is training or not.
        """

        if self.__input_shape is None:
            raise ValueError("GRU has no input shape; add an Input layer first.")
        if inputs.ndim != 3 or inputs.shape[2] != self.__input_shape[1]:
            raise ValueError(
                f"GRU expects inputs of shape (N, T, {self.__input_shape[1]}), "
                f"got {inputs.shape}.")

        N, T, _ = inputs.shape
        U = self.__units

        h = np.zeros((N, U))
        z_seq = np.empty((N, T, U))
        r_seq = np.empty((N, T, U))
        h_tilde_seq = np.empty((N, T, U))
        h_seq = np.empty((N, T, U))

        for t in range(T):
            # z and r share the full pre-activation ...
            pre = inputs[:, t, :] @ self.params["W_xh"] + h @ self.params["W_hh"]
            if "b" in self.params:
                pre = pre + self.params["b"]
            z = self.__activation_mapper[self.__recurrent_activation](
                pre[:, :U], **self.__recurrent_activation_config)
            r = self.__activation_mapper[self.__recurrent_activation](
                pre[:, U:2 * U], **self.__recurrent_activation_config)

            # ... while the candidate sees the old state through the reset
            # gate: (r * h_{t-1}) instead of h_{t-1}.
            cand = inputs[:, t, :] @ self.params["W_xh"][:, 2 * U:] \
                + (r * h) @ self.params["W_hh"][:, 2 * U:]
            if "b" in self.params:
                cand = cand + self.params["b"][2 * U:]
            h_tilde = self.__activation_mapper[self.__activation](cand, **self.__activation_config)

            h = (1 - z) * h + z * h_tilde

            z_seq[:, t, :] = z
            r_seq[:, t, :] = r
            h_tilde_seq[:, t, :] = h_tilde
            h_seq[:, t, :] = h

        self.inputs = inputs
        self.__z_seq = z_seq
        self.__r_seq = r_seq
        self.__h_tilde_seq = h_tilde_seq
        self.__h_seq = h_seq

        if self.__return_sequences:
            return h_seq
        return h

    def backward(
            self,
            grad: np.ndarray,
        ) -> np.ndarray:

        """
        Backward propagation: BPTT through the update, reset and candidate
        paths.

        The gradient w.r.t. h_{t-1} is the sum of four terms: the direct
        copy path (1 - z_t), the candidate path gated by the reset gate
        r_t, and the two gate paths through z_t and r_t.

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

        dX = np.empty_like(self.inputs)
        dh = np.zeros((N, U))

        for t in range(T - 1, -1, -1):
            dh = dh + d_out[:, t, :]
            z = self.__z_seq[:, t, :]
            r = self.__r_seq[:, t, :]
            h_tilde = self.__h_tilde_seq[:, t, :]
            # h_{t-1}; the initial state h_{-1} is zeros
            h_prev = np.zeros((N, U)) if t == 0 else self.__h_seq[:, t - 1, :]

            # h_t = (1 - z) * h_{t-1} + z * h̃
            dz = dh * (h_tilde - h_prev) * self.__gate_deriv(z, **self.__recurrent_activation_config)
            dh_tilde = dh * z * self.__cand_deriv(h_tilde, **self.__activation_config)

            # h̃ = tanh(x_t @ W_xh + (r * h_{t-1}) @ W_hh + b_h)
            r_h = r * h_prev
            dr = (dh_tilde @ self.params["W_hh"][:, 2 * U:].T) * h_prev * self.__gate_deriv(r, **self.__recurrent_activation_config)
            dh_prev = dh * (1 - z) \
                + (dh_tilde @ self.params["W_hh"][:, 2 * U:].T) * r \
                + dz @ self.params["W_hh"][:, :U].T \
                + dr @ self.params["W_hh"][:, U:2 * U].T

            x_t = self.inputs[:, t, :]
            self.grads["W_xh"][:, :U] += x_t.T @ dz
            self.grads["W_xh"][:, U:2 * U] += x_t.T @ dr
            self.grads["W_xh"][:, 2 * U:] += x_t.T @ dh_tilde
            self.grads["W_hh"][:, :U] += h_prev.T @ dz
            self.grads["W_hh"][:, U:2 * U] += h_prev.T @ dr
            self.grads["W_hh"][:, 2 * U:] += r_h.T @ dh_tilde
            if "b" in self.grads:
                self.grads["b"][:U] += dz.sum(axis=0)
                self.grads["b"][U:2 * U] += dr.sum(axis=0)
                self.grads["b"][2 * U:] += dh_tilde.sum(axis=0)
            dX[:, t, :] = dh_tilde @ self.params["W_xh"][:, 2 * U:].T \
                + dz @ self.params["W_xh"][:, :U].T \
                + dr @ self.params["W_xh"][:, U:2 * U].T

            dh = dh_prev

        if self.__activation_deriv:
            dX *= self.__activation_deriv(self.inputs, **self.__activation_derive_config)
        return dX

    @property
    def units(self):
        return self.__units

    @property
    def activation(self):
        # Deliberately None: h_t = (1 - z) * h_{t-1} + z * h̃ is not the
        # elementwise activation of a single pre-activation, so the generic
        # activation-derivative chain must skip this layer; the output
        # chain is handled inside backward.
        return None

    @property
    def activation_config(self):
        return self.__activation_config

    @property
    def return_sequences(self):
        return self.__return_sequences

    @property
    def output_dim(self):
        return int(np.prod(self.__output_shape))

    @property
    def output_shape(self):
        return self.__output_shape

    def __str__(self):
        return f"GRU(units={self.__units}, return_sequences={self.__return_sequences})"
