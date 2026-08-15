"""Unit tests for the recurrent layers: SimpleRNN, LSTM and GRU.

The layers are pinned against per-timestep loop reference implementations
(the plain definition of each recurrence), plus whole-model finite-difference
gradient checks and an end-to-end training test on MNIST read row by row.
"""
import csv
import os

import numpy as np
import pytest

from numpy_keras import Sequential
from numpy_keras import layers
from numpy_keras.activations import functional as F

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "mnist_train_small.csv")


# ---------------------------------------------------------------------------
# Reference implementations (per-timestep loops -- the teaching definition)
# ---------------------------------------------------------------------------

def rnn_forward_ref(X, W_xh, W_hh, b, act, return_sequences):
    """SimpleRNN: h_t = act(x_t @ W_xh + h_{t-1} @ W_hh + b)."""
    N, T, _ = X.shape
    U = W_hh.shape[0]
    h = np.zeros((N, U))
    outs = []
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        h = act(pre)
        outs.append(h)
    return np.stack(outs, axis=1) if return_sequences else h


def rnn_backward_ref(X, W_xh, W_hh, b, act, act_deriv, grad, return_sequences):
    """SimpleRNN BPTT: unroll the chain rule backwards over the timesteps."""
    N, T, F = X.shape
    U = W_hh.shape[0]
    # recompute the forward pass to get the post-activation states
    h = np.zeros((N, U))
    hs = [h]
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        h = act(pre)
        hs.append(h)

    d_out = np.zeros((N, T, U))
    if return_sequences:
        d_out = grad
    else:
        d_out[:, -1, :] = grad

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros(U)
    dX = np.zeros_like(X)
    dh = np.zeros((N, U))
    for t in range(T - 1, -1, -1):
        dh = dh + d_out[:, t, :]
        d_pre = dh * act_deriv(hs[t + 1])      # derivs take post-activation values
        dW_hh = dW_hh + hs[t].T @ d_pre
        dW_xh = dW_xh + X[:, t, :].T @ d_pre
        db = db + d_pre.sum(axis=0)
        dX[:, t, :] = d_pre @ W_xh.T
        dh = d_pre @ W_hh.T
    return dW_xh, dW_hh, db, dX


def lstm_forward_ref(X, W_xh, W_hh, b, return_sequences):
    """LSTM with gates [i, f, g, o]: c_t = f*c_{t-1} + i*g; h_t = o*tanh(c_t)."""
    N, T, _ = X.shape
    U = W_hh.shape[0]
    h = np.zeros((N, U))
    c = np.zeros((N, U))
    outs = []
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        i = F.sigmoid(pre[:, :U])
        f = F.sigmoid(pre[:, U:2 * U])
        g = F.tanh(pre[:, 2 * U:3 * U])
        o = F.sigmoid(pre[:, 3 * U:])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs.append(h)
    return np.stack(outs, axis=1) if return_sequences else h


def lstm_backward_ref(X, W_xh, W_hh, b, grad, return_sequences):
    """LSTM BPTT: the cell gradient receives the output path plus the
    future, gated by the forget gate f_{t+1}."""
    N, T, _ = X.shape
    U = W_hh.shape[0]
    i_seq = np.empty((N, T, U))
    f_seq = np.empty((N, T, U))
    g_seq = np.empty((N, T, U))
    o_seq = np.empty((N, T, U))
    c_seq = np.empty((N, T, U))
    h_seq = np.empty((N, T, U))
    h = np.zeros((N, U))
    c = np.zeros((N, U))
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        i = F.sigmoid(pre[:, :U])
        f = F.sigmoid(pre[:, U:2 * U])
        g = F.tanh(pre[:, 2 * U:3 * U])
        o = F.sigmoid(pre[:, 3 * U:])
        c = f * c + i * g
        h = o * np.tanh(c)
        i_seq[:, t, :] = i
        f_seq[:, t, :] = f
        g_seq[:, t, :] = g
        o_seq[:, t, :] = o
        c_seq[:, t, :] = c
        h_seq[:, t, :] = h

    d_out = np.zeros((N, T, U))
    if return_sequences:
        d_out = grad
    else:
        d_out[:, -1, :] = grad

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros(4 * U)
    dX = np.zeros_like(X)
    dh = np.zeros((N, U))
    dc = np.zeros((N, U))
    for t in range(T - 1, -1, -1):
        dh = dh + d_out[:, t, :]
        tanh_c = np.tanh(c_seq[:, t, :])
        h_prev = np.zeros((N, U)) if t == 0 else h_seq[:, t - 1, :]
        c_prev = np.zeros((N, U)) if t == 0 else c_seq[:, t - 1, :]
        d_o = dh * tanh_c * F.sigmoid_deriv(o_seq[:, t, :])
        d_c = dh * o_seq[:, t, :] * F.tanh_deriv(tanh_c)
        if t < T - 1:
            d_c = d_c + dc * f_seq[:, t + 1, :]
        d_i = d_c * g_seq[:, t, :] * F.sigmoid_deriv(i_seq[:, t, :])
        d_f = d_c * c_prev * F.sigmoid_deriv(f_seq[:, t, :])
        d_g = d_c * i_seq[:, t, :] * F.tanh_deriv(g_seq[:, t, :])
        d_pre = np.concatenate([d_i, d_f, d_g, d_o], axis=1)
        dW_hh = dW_hh + h_prev.T @ d_pre
        dW_xh = dW_xh + X[:, t, :].T @ d_pre
        db = db + d_pre.sum(axis=0)
        dX[:, t, :] = d_pre @ W_xh.T
        dh = d_pre @ W_hh.T
        dc = d_c
    return dW_xh, dW_hh, db, dX


def gru_forward_ref(X, W_xh, W_hh, b, return_sequences):
    """GRU (reset_after=False): h_t = (1-z)*h_{t-1} + z*tanh(x + (r*h)@W)."""
    N, T, _ = X.shape
    U = W_hh.shape[0]
    h = np.zeros((N, U))
    outs = []
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        z = F.sigmoid(pre[:, :U])
        r = F.sigmoid(pre[:, U:2 * U])
        cand = X[:, t, :] @ W_xh[:, 2 * U:] + (r * h) @ W_hh[:, 2 * U:]
        if b is not None:
            cand = cand + b[2 * U:]
        h_tilde = F.tanh(cand)
        h = (1 - z) * h + z * h_tilde
        outs.append(h)
    return np.stack(outs, axis=1) if return_sequences else h


def gru_backward_ref(X, W_xh, W_hh, b, grad, return_sequences):
    """GRU BPTT: the h_{t-1} gradient is the copy path plus the candidate
    path through the reset gate plus the two gate paths."""
    N, T, _ = X.shape
    U = W_hh.shape[0]
    z_seq = np.empty((N, T, U))
    r_seq = np.empty((N, T, U))
    h_tilde_seq = np.empty((N, T, U))
    h_seq = np.empty((N, T, U))
    h = np.zeros((N, U))
    for t in range(T):
        pre = X[:, t, :] @ W_xh + h @ W_hh
        if b is not None:
            pre = pre + b
        z = F.sigmoid(pre[:, :U])
        r = F.sigmoid(pre[:, U:2 * U])
        cand = X[:, t, :] @ W_xh[:, 2 * U:] + (r * h) @ W_hh[:, 2 * U:]
        if b is not None:
            cand = cand + b[2 * U:]
        h_tilde = F.tanh(cand)
        h = (1 - z) * h + z * h_tilde
        z_seq[:, t, :] = z
        r_seq[:, t, :] = r
        h_tilde_seq[:, t, :] = h_tilde
        h_seq[:, t, :] = h

    d_out = np.zeros((N, T, U))
    if return_sequences:
        d_out = grad
    else:
        d_out[:, -1, :] = grad

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros(3 * U)
    dX = np.zeros_like(X)
    dh = np.zeros((N, U))
    for t in range(T - 1, -1, -1):
        dh = dh + d_out[:, t, :]
        z = z_seq[:, t, :]
        r = r_seq[:, t, :]
        h_tilde = h_tilde_seq[:, t, :]
        h_prev = np.zeros((N, U)) if t == 0 else h_seq[:, t - 1, :]
        dz = dh * (h_tilde - h_prev) * F.sigmoid_deriv(z)
        dh_tilde = dh * z * F.tanh_deriv(h_tilde)
        r_h = r * h_prev
        dr = (dh_tilde @ W_hh[:, 2 * U:].T) * h_prev * F.sigmoid_deriv(r)
        dh_prev = dh * (1 - z) \
            + (dh_tilde @ W_hh[:, 2 * U:].T) * r \
            + dz @ W_hh[:, :U].T \
            + dr @ W_hh[:, U:2 * U].T
        x_t = X[:, t, :]
        dW_xh[:, :U] += x_t.T @ dz
        dW_xh[:, U:2 * U] += x_t.T @ dr
        dW_xh[:, 2 * U:] += x_t.T @ dh_tilde
        dW_hh[:, :U] += h_prev.T @ dz
        dW_hh[:, U:2 * U] += h_prev.T @ dr
        dW_hh[:, 2 * U:] += r_h.T @ dh_tilde
        db[:U] += dz.sum(axis=0)
        db[U:2 * U] += dr.sum(axis=0)
        db[2 * U:] += dh_tilde.sum(axis=0)
        dX[:, t, :] = dh_tilde @ W_xh[:, 2 * U:].T \
            + dz @ W_xh[:, :U].T \
            + dr @ W_xh[:, U:2 * U].T
        dh = dh_prev
    return dW_xh, dW_hh, db, dX


# ---------------------------------------------------------------------------
# Parity: layer vs loop reference
# ---------------------------------------------------------------------------

def make_rnn(layer_cls, units=4, return_sequences=False, use_bias=True,
             input_shape=(5, 3), **kwargs):
    layer = layer_cls(units=units, return_sequences=return_sequences,
                      use_bias=use_bias, **kwargs)
    layer.set_input_shape(input_shape)
    return layer


def randomize_rnn(layer, seed):
    rng = np.random.RandomState(seed)
    layer.params["W_xh"] = rng.randn(*layer.params["W_xh"].shape)
    layer.params["W_hh"] = rng.randn(*layer.params["W_hh"].shape)
    if "b" in layer.params:
        layer.params["b"] = rng.randn(*layer.params["b"].shape)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_simple_rnn_forward_matches_reference(return_sequences, use_bias, activation):
    layer = make_rnn(layers.SimpleRNN, return_sequences=return_sequences,
                     use_bias=use_bias, activation=activation)
    randomize_rnn(layer, seed=0)
    X = np.random.RandomState(1).randn(4, 5, 3)
    got = layer.forward(X, is_training=True)
    act = F.tanh if activation == "tanh" else F.relu
    want = rnn_forward_ref(X, layer.params["W_xh"], layer.params["W_hh"],
                           layer.params.get("b"), act, return_sequences)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_simple_rnn_backward_matches_reference(return_sequences, use_bias, activation):
    layer = make_rnn(layers.SimpleRNN, return_sequences=return_sequences,
                     use_bias=use_bias, activation=activation)
    randomize_rnn(layer, seed=2)
    X = np.random.RandomState(3).randn(4, 5, 3)
    layer.forward(X, is_training=True)
    grad = np.random.RandomState(4).randn(4, 5, 4) if return_sequences \
        else np.random.RandomState(4).randn(4, 4)
    grad_next = layer.backward(grad)

    act = F.tanh if activation == "tanh" else F.relu
    deriv = F.tanh_deriv if activation == "tanh" else F.relu_deriv
    dW_xh, dW_hh, db, dX = rnn_backward_ref(
        X, layer.params["W_xh"], layer.params["W_hh"], layer.params.get("b"),
        act, deriv, grad, return_sequences)
    np.testing.assert_allclose(layer.grads["W_xh"], dW_xh, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(layer.grads["W_hh"], dW_hh, rtol=1e-12, atol=1e-12)
    if use_bias:
        np.testing.assert_allclose(layer.grads["b"], db, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_next, dX, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
def test_lstm_forward_matches_reference(return_sequences, use_bias):
    layer = make_rnn(layers.LSTM, return_sequences=return_sequences, use_bias=use_bias)
    randomize_rnn(layer, seed=5)
    X = np.random.RandomState(6).randn(4, 5, 3)
    got = layer.forward(X, is_training=True)
    want = lstm_forward_ref(X, layer.params["W_xh"], layer.params["W_hh"],
                            layer.params.get("b"), return_sequences)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
def test_lstm_backward_matches_reference(return_sequences, use_bias):
    layer = make_rnn(layers.LSTM, return_sequences=return_sequences, use_bias=use_bias)
    randomize_rnn(layer, seed=7)
    X = np.random.RandomState(8).randn(4, 5, 3)
    layer.forward(X, is_training=True)
    grad = np.random.RandomState(9).randn(4, 5, 4) if return_sequences \
        else np.random.RandomState(9).randn(4, 4)
    grad_next = layer.backward(grad)

    dW_xh, dW_hh, db, dX = lstm_backward_ref(
        X, layer.params["W_xh"], layer.params["W_hh"], layer.params.get("b"),
        grad, return_sequences)
    np.testing.assert_allclose(layer.grads["W_xh"], dW_xh, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(layer.grads["W_hh"], dW_hh, rtol=1e-12, atol=1e-12)
    if use_bias:
        np.testing.assert_allclose(layer.grads["b"], db, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_next, dX, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
def test_gru_forward_matches_reference(return_sequences, use_bias):
    layer = make_rnn(layers.GRU, return_sequences=return_sequences, use_bias=use_bias)
    randomize_rnn(layer, seed=10)
    X = np.random.RandomState(11).randn(4, 5, 3)
    got = layer.forward(X, is_training=True)
    want = gru_forward_ref(X, layer.params["W_xh"], layer.params["W_hh"],
                           layer.params.get("b"), return_sequences)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("use_bias", [True, False])
def test_gru_backward_matches_reference(return_sequences, use_bias):
    layer = make_rnn(layers.GRU, return_sequences=return_sequences, use_bias=use_bias)
    randomize_rnn(layer, seed=12)
    X = np.random.RandomState(13).randn(4, 5, 3)
    layer.forward(X, is_training=True)
    grad = np.random.RandomState(14).randn(4, 5, 4) if return_sequences \
        else np.random.RandomState(14).randn(4, 4)
    grad_next = layer.backward(grad)

    dW_xh, dW_hh, db, dX = gru_backward_ref(
        X, layer.params["W_xh"], layer.params["W_hh"], layer.params.get("b"),
        grad, return_sequences)
    np.testing.assert_allclose(layer.grads["W_xh"], dW_xh, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(layer.grads["W_hh"], dW_hh, rtol=1e-12, atol=1e-12)
    if use_bias:
        np.testing.assert_allclose(layer.grads["b"], db, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_next, dX, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# return_sequences semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_cls", [layers.SimpleRNN, layers.LSTM, layers.GRU])
def test_last_timestep_gradient_equals_full_sequence_backward(layer_cls):
    """With return_sequences=False the gradient arrives only at h_T; the
    resulting dX must equal the return_sequences=True backward with a
    gradient that is zero everywhere except the last timestep."""
    X = np.random.RandomState(15).randn(4, 5, 3)
    G = np.random.RandomState(16).randn(4, 4)
    last = make_rnn(layer_cls, return_sequences=False)
    full = make_rnn(layer_cls, return_sequences=True)
    randomize_rnn(last, seed=17)
    for k in full.params:
        full.params[k] = last.params[k].copy()
    last.forward(X, is_training=True)
    full.forward(X, is_training=True)
    dX_last = last.backward(G)
    d_out = np.zeros((4, 5, 4))
    d_out[:, -1, :] = G
    dX_full = full.backward(d_out)
    np.testing.assert_allclose(dX_last, dX_full, rtol=1e-12, atol=1e-12)


def test_rnn_output_shapes():
    for cls in (layers.SimpleRNN, layers.LSTM, layers.GRU):
        last = make_rnn(cls, units=6, return_sequences=False, input_shape=(9, 4))
        assert last.output_shape == (6,)
        assert last.output_dim == 6
        full = make_rnn(cls, units=7, return_sequences=True, input_shape=(9, 4))
        assert full.output_shape == (9, 7)
        assert full.output_dim == 63


# ---------------------------------------------------------------------------
# Shape chain
# ---------------------------------------------------------------------------

def test_stacked_rnns_build_and_share_the_sequence_axis():
    model = Sequential()
    model.add(layers.Input((7, 4)))
    model.add(layers.LSTM(6, return_sequences=True))
    model.add(layers.GRU(5))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="mse", optimizer="sgd")
    lstm, gru, dense = (model.layers["lstm_1"], model.layers["gru_1"],
                        model.layers["dense_1"])
    assert lstm.output_shape == (7, 6)
    assert gru.output_shape == (5,)
    assert gru.params["W_xh"].shape == (6, 15)
    assert dense.params["W"].shape == (5, 2)


def test_rnn_flatten_dense_chain():
    model = Sequential()
    model.add(layers.Input((5, 4)))
    model.add(layers.GRU(3, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    assert model.layers["flatten_1"].output_dim == 15
    assert model.layers["dense_1"].params["W"].shape == (15, 2)


def test_batch_norm_between_rnns():
    model = Sequential()
    model.add(layers.Input((5, 4)))
    model.add(layers.SimpleRNN(3, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(2))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd")
    assert model.layers["batch_normalization_1"].params["gamma"].shape == (3,)


def test_rnn_summary_prints_sequence_shapes(capsys):
    model = Sequential()
    model.add(layers.Input((7, 4)))
    model.add(layers.LSTM(6, return_sequences=True))
    model.add(layers.GRU(5))
    model.add(layers.Dense(2, activation="softmax"))
    model.summary()
    out = capsys.readouterr().out
    assert "(7, 6)" in out
    assert "(5,)" in out


# ---------------------------------------------------------------------------
# The generic activation-derivative chain
# ---------------------------------------------------------------------------

def test_simple_rnn_backward_applies_previous_activation_deriv():
    """SimpleRNN participates in the generic chain like Dense: the returned
    dX is multiplied by the previous layer's activation deriv at its input."""
    layer = make_rnn(layers.SimpleRNN)
    randomize_rnn(layer, seed=18)
    X = np.random.RandomState(19).randn(4, 5, 3)
    layer.forward(X, is_training=True)
    grad = np.random.RandomState(20).randn(4, 4)
    plain = layer.backward(grad)
    layer.set_activation_deriv("tanh", {})
    chained = layer.backward(grad)
    np.testing.assert_allclose(chained, plain * F.tanh_deriv(X), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("layer_cls", [layers.SimpleRNN, layers.LSTM, layers.GRU])
def test_rnn_layers_reset_the_chain_for_the_next_layer(layer_cls):
    """RNN layers own their output chain (it runs inside backward, since
    the hidden state also feeds the recurrence), so the next layer must
    not apply any deriv: their activation property is None."""
    model = Sequential()
    model.add(layers.Input((5, 3)))
    model.add(layer_cls(4))
    model.add(layers.Dense(1, activation="linear"))
    dense = model.layers["dense_1"]
    rnn = next(layer for layer in model.layers.values() if isinstance(layer, layer_cls))
    assert rnn.activation is None
    assert dense._Dense__activation_deriv is None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_cls", [layers.SimpleRNN, layers.LSTM, layers.GRU])
def test_rnn_rejects_non_sequence_input_shapes(layer_cls):
    with pytest.raises(ValueError):
        layer_cls(3).set_input_shape((12,))          # 1D, e.g. after Flatten
    with pytest.raises(ValueError):
        layer_cls(3).set_input_shape((4, 4, 2))      # 3D, e.g. after Conv2D


@pytest.mark.parametrize("layer_cls", [layers.SimpleRNN, layers.LSTM, layers.GRU])
def test_rnn_raises_without_input_shape(layer_cls):
    with pytest.raises(ValueError):
        layer_cls(3).forward(np.random.randn(4, 5, 2), is_training=True)


def test_rnn_rejects_wrong_ndim_or_features():
    layer = make_rnn(layers.GRU, input_shape=(5, 3))
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 5), is_training=True)
    with pytest.raises(ValueError):
        layer.forward(np.random.randn(4, 5, 4), is_training=True)  # wrong features


def test_rnn_rejects_softmax_activation():
    with pytest.raises(ValueError):
        layers.SimpleRNN(3, activation="softmax")
    with pytest.raises(ValueError):
        layers.LSTM(3, activation="softmax")
    with pytest.raises(ValueError):
        layers.GRU(3, recurrent_activation="softmax")


def test_dense_after_sequence_rnn_requires_flatten():
    model = Sequential()
    model.add(layers.Input((5, 3)))
    model.add(layers.LSTM(4, return_sequences=True))
    with pytest.raises(ValueError, match="Flatten"):
        model.add(layers.Dense(1))


# ---------------------------------------------------------------------------
# Whole-model finite-difference gradient checks
# ---------------------------------------------------------------------------

def _gradient_check_model(model, X, y, seed):
    """Backprop grads of every parameter must equal finite differences of
    the reported loss."""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    model.compile(loss="mse", optimizer="sgd")
    for layer in model.layers.values():
        if hasattr(layer, "params"):
            for k in layer.params:
                layer.params[k] = rng.randn(*layer.params[k].shape) * 0.3

    y_hat = model._Sequential__forward(X, is_training=True)
    _, grad = model._Sequential__criterion(y, y_hat)
    model._Sequential__backward(grad)

    eps = 1e-6
    for name, layer in model.layers.items():
        if not hasattr(layer, "params"):
            continue
        for k, p in layer.params.items():
            numerical = np.zeros_like(p.ravel())
            for i in range(p.size):
                orig = p.ravel()[i]
                p.ravel()[i] = orig + eps
                l1 = model._Sequential__loss_func(y, model._Sequential__forward(X, is_training=False))
                p.ravel()[i] = orig - eps
                l2 = model._Sequential__loss_func(y, model._Sequential__forward(X, is_training=False))
                p.ravel()[i] = orig
                numerical[i] = (l1 - l2) / (2 * eps)
            np.testing.assert_allclose(
                layer.grads[k].ravel(), numerical, rtol=1e-3, atol=1e-6,
                err_msg=f"{name}.{k}")


def _make_sequence_model(*rnn_layers):
    model = Sequential()
    model.add(layers.Input((5, 3)))
    for layer in rnn_layers:
        model.add(layer)
    return model


def test_simple_rnn_model_gradient_check():
    rng = np.random.RandomState(21)
    model = _make_sequence_model(layers.SimpleRNN(4), layers.Dense(1, activation="linear"))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 1), seed=21)


def test_lstm_model_gradient_check():
    rng = np.random.RandomState(22)
    model = _make_sequence_model(layers.LSTM(4), layers.Dense(1, activation="linear"))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 1), seed=22)


def test_gru_model_gradient_check():
    rng = np.random.RandomState(23)
    model = _make_sequence_model(layers.GRU(4), layers.Dense(1, activation="linear"))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 1), seed=23)


def test_stacked_lstm_model_gradient_check():
    rng = np.random.RandomState(24)
    model = _make_sequence_model(layers.LSTM(4, return_sequences=True),
                                 layers.LSTM(3),
                                 layers.Dense(1, activation="linear"))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 1), seed=24)


def test_gru_flatten_model_gradient_check():
    rng = np.random.RandomState(25)
    model = _make_sequence_model(layers.GRU(4, return_sequences=True),
                                 layers.Flatten(),
                                 layers.Dense(1, activation="linear"))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 1), seed=25)


def test_simple_rnn_as_last_layer_gradient_check():
    """SimpleRNN as the last layer: the criterion applies tanh_deriv at the
    output (the generic chain), and backward handles the recurrence."""
    rng = np.random.RandomState(26)
    model = _make_sequence_model(layers.SimpleRNN(4))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 4), seed=26)


@pytest.mark.parametrize("layer_cls", [layers.LSTM, layers.GRU])
def test_lstm_gru_as_last_layer_gradient_check(layer_cls):
    """LSTM/GRU as the last layer: the criterion must skip the generic
    deriv chain (activation is None); the internal chain is in backward."""
    rng = np.random.RandomState(27)
    model = _make_sequence_model(layer_cls(4))
    _gradient_check_model(model, rng.randn(4, 5, 3), rng.randn(4, 4), seed=27)


def test_lstm_as_last_layer_fits_without_error():
    np.random.seed(28)
    rng = np.random.RandomState(28)
    X = rng.randn(24, 4, 3)
    y = rng.randn(24, 5)
    model = _make_sequence_model(layers.LSTM(5))
    model.compile(loss="mse", optimizer="sgd")
    history = model.fit(X, y, batch_size=8, epochs=2)
    assert len(history.loss) == 2


# ---------------------------------------------------------------------------
# End to end: MNIST read row by row
# ---------------------------------------------------------------------------

def _load_mnist_rows(n):
    with open(DATA_PATH) as f:
        rows = list(csv.reader(f))
    X = np.array([[float(v) for v in r[1:]] for r in rows]) / 255.0
    y = np.array([int(r[0]) for r in rows])
    # each 28x28 image becomes 28 timesteps of 28 pixels (row by row)
    return X.reshape(-1, 28, 28)[:n], y[:n]


def test_lstm_end_to_end_classifies_mnist_rows():
    np.random.seed(29)
    X, y = _load_mnist_rows(800)
    model = Sequential()
    model.add(layers.Input((28, 28)))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.optimizer.learning_rate = 0.01

    history = model.fit(X, y, batch_size=32, epochs=10, shuffle=True)
    assert history.metrics["train_accuracy"][-1] > 0.75
    pred = model.predict(X)
    assert pred.shape == (800,)
    assert np.mean(pred == y) > 0.75
