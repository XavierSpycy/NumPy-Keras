from collections import defaultdict
from typing import (
    List, 
    Tuple,
    Literal,
    Union,
    Optional,
    Sequence,
)

import numpy as np

from . import utils
from .. import (
    callbacks,
    layers,
    losses,
    metrics,
    optimizers,
)
from ..cython import _kernels as _ck
from .. import backend as _backend  # module: _backend.xp follows set_backend
from ..backend import (
    asarray,
    asnumpy,
    is_cupy_array,
    is_numpy_array,
    item,
    on_gpu,
)

class Sequential:
    __idx2label = None

    def __init__(
            self,
            layers: List[
                Union[
                    layers.Activation,
                    layers.BatchNormalization,
                    layers.Conv2D,
                    layers.Dense,
                    layers.Dropout,
                    layers.Flatten,
                    layers.GRU,
                    layers.Input,
                    layers.LSTM,
                    layers.MaxPool2D,
                    layers.SimpleRNN,
                ]
            ] = [],
            dtype=None,
        ) -> None:

        """
        Parameters:
        - layers (list, optional): Layers of the model.
        - dtype (optional): Compute dtype of the model, e.g. numpy.float32.
          Default is None (float64, the library's historical default).
          Parameters and per-batch inputs are cast to this dtype at build
          / forward time; float32 saves memory and is the fast path on
          GPUs (the Cython kernels stay float64-only and simply skip
          float32 models).
        """

        self.__layer_counter = defaultdict(int)
        self.__layers = {}
        self.__dtype = None if dtype is None else np.dtype(dtype)
        for layer in layers:
            self.__layer_counter[layer.__class__.__name__] += 1
            layer_index = f"{utils.camel_to_snake(layer.__class__.__name__)}_{self.__layer_counter[layer.__class__.__name__]}"
            self.__layers[layer_index] = layer
        self.__build()

    def add(
            self, 
            layer,
            rebuild: bool = True,
        ) -> None:
        self.__layer_counter[layer.__class__.__name__] += 1
        self.__layers[f"{utils.camel_to_snake(layer.__class__.__name__)}_{self.__layer_counter[layer.__class__.__name__]}"] = layer
        if rebuild:
            self.__build()
    
    def compile(
            self, 
            loss: Union[
                Literal[
                    'mse', 
                    'categorical_crossentropy', 
                    'sparse_categorical_crossentropy',
                ], 
                losses.CategoricalCrossEntropy, 
                losses.MSE,
            ] = 'mse', 
            optimizer: Union[
                Literal[
                    'adadelta', 
                    'adagrad', 
                    'adam', 
                    'sgd',
                ], 
                optimizers.Adadelta, 
                optimizers.Adagrad, 
                optimizers.Adam, 
                optimizers.SGD,
            ] = 'sgd',
            metrics: List = [],
        ) -> None:
        
        self.__history = callbacks.History()

        if isinstance(loss, str):
            self.__loss_func = losses._LossMapper()[loss]
        else:
            self.__loss_func = loss

        if isinstance(optimizer, str):
            self.optimizer = optimizers._OptimMapper()[optimizer]
        else:
            self.optimizer = optimizer
        
        self.__metrics = metrics
    
    def evaluate(
            self,
            X: Sequence[np.float64],
            y: Sequence[np.float64],
            batch_size: int = 32,
        ) -> List[np.float64]:

        self.__sync_backend()
        X = asnumpy(X) if is_cupy_array(X) else np.array(X).copy()
        y = asnumpy(y) if is_cupy_array(y) else np.array(y).copy()

        y_hat = self.__forward(X, is_training=False)
        if self.__loss_func.name == 'sparse_categorical_crossentropy':
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            y_, _ = utils.one_hot_encode(y, self.__idx2label)
        else:
            y_ = y
        loss = item(self.__loss_func(asarray(y_), y_hat))

        if self.__metrics:
            y_pred = self.predict(X, batch_size)
            return metrics._MetricMapper()[self.__metrics[0]](y, y_pred)
        else:
            return loss
    
    def fit(
            self, 
            X: Sequence[np.float64], 
            y: Sequence[np.float64], 
            batch_size: int = 32, 
            epochs: int = 1,
            verbose: int = 0,
            callbacks: Optional[List[Union[callbacks.EarlyStopping, callbacks.lr_scheduler.LRScheduler]]] = None,
            validation_split: float = 0.0,
            validation_data: Optional[Tuple[Sequence[np.float64], Sequence[np.float64]]] = None,
            shuffle: bool = True, 
            initial_epoch: int = 0,
            steps_per_epoch: Optional[int] = None,
            validation_batch_size: Optional[int] = None,
            validation_freq: int = 1,
        ) -> callbacks.History:

        self.__sync_backend()
        X_train = asnumpy(X) if is_cupy_array(X) else np.array(X)
        y_train = asnumpy(y) if is_cupy_array(y) else np.array(y)
        X_test = None
        y_test = None

        self.stop_training = False
        self.__best_weights = None
        
        if self.__loss_func.name == 'sparse_categorical_crossentropy':
            if y_train.ndim == 2 and y_train.shape[1] == 1:
                y_train = y_train.ravel()
            y_train, self.__idx2label = utils.one_hot_encode(y_train)

        if validation_data is not None:
            X_test, y_test = validation_data
            if self.__loss_func.name == 'sparse_categorical_crossentropy':
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.ravel()
                y_test, _ = utils.one_hot_encode(y_test, self.__idx2label)
        elif not np.isclose(validation_split, 0.0):
            X_train, X_test, y_train, y_test = utils.train_test_split(
                X_train, y_train, test_size=validation_split)

        for callback in callbacks or []:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(
                    model=self,
                )
        
        for epoch in utils.conditional_tqdm(range(initial_epoch, epochs), verbose == 1):
            indices = np.arange(X_train.shape[0])
            if shuffle:
                np.random.shuffle(indices)
            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            loss = np.zeros(len(batches))
            step = 0
            for batch in utils.conditional_tqdm(batches, verbose == 2):
                X_batch = X_train[batch]
                y_batch = y_train[batch]
                y_hat = self.__forward(X_batch, is_training=True)
                loss[step], grad = self.__criterion(y_batch, y_hat)
                self.__backward(grad)
                self.optimizer.update(self.layers.values())
                step += 1
                if steps_per_epoch and step >= steps_per_epoch:
                    break
            
            if epoch % validation_freq == 0:
                # Full-data predictions only feed the metric loop; without
                # metrics the extra forward passes are pure waste.
                if self.__metrics:
                    y_pred_train = self.predict(
                        X_train,
                        batch_size=validation_batch_size if validation_batch_size is not None else batch_size)
                    y_pred_test = self.predict(
                        X_test,
                        batch_size=validation_batch_size if validation_batch_size is not None else batch_size
                    ) if X_test is not None else None
                else:
                    y_pred_train = None
                    y_pred_test = None

                if X_test is not None:
                    if not "val_loss" in self.__history.metrics:
                        self.__history.metrics["val_loss"] = []
                    self.__history.metrics["val_loss"].append(
                        item(self.__loss_func(asarray(y_test), self.__forward(X_test, is_training=False)))
                    )

                for metric in self.__metrics:
                    if not "train_" + metric in self.__history.metrics:
                        self.__history.metrics["train_" + metric] = []
                    self.__history.metrics["train_" + metric].append(
                        metrics._MetricMapper()[metric](
                            y_train if self.__idx2label is None else utils.one_hot_decode(y_train, self.__idx2label), 
                            y_pred_train
                        )
                    )
                    if y_pred_test is not None:
                        if not "val_" + metric in self.__history.metrics:
                            self.__history.metrics["val_" + metric] = []
                        self.__history.metrics["val_" + metric].append(
                            metrics._MetricMapper()[metric](
                                y_test if self.__idx2label is None else utils.one_hot_decode(y_test, self.__idx2label), 
                                y_pred_test
                            )
                        )
                self.__history.validation_epochs.append(epoch)
            self.__history.loss.append(np.mean(loss))
            for callback in callbacks or []:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(
                        model=self,
                    )

                    if hasattr(callback, 'save_best') and callback.save_best:
                        # copy the arrays: optimizers update params in place,
                        # so keeping references would alias later epochs
                        self.__best_weights = {
                            idx: {param: value.copy() for param, value in layer.params.items()}
                            for idx, layer in self.layers.items() if hasattr(layer, 'params')
                        }

            if self.stop_training:
                break

        if self.__best_weights is not None:
            for idx, params in self.__best_weights.items():
                for param, value in params.items():
                    self.__layers[idx].params[param] = value

        return self.history

    def pop(
            self, 
            rebuild: bool = True,
        ) -> None:
        
        self.__layers.popitem()
        if rebuild:
            self.__build()
    
    def predict(
            self, 
            X: Sequence[np.float64], 
            batch_size: int = 32,
        ) -> np.ndarray:
        
        """
        Predict on the given data.

        The return type follows the model's task: with a
        ``sparse_categorical_crossentropy`` loss the outputs are decoded
        into integer class labels (an (N,) array via ``idx2label``); with
        any other loss the raw network outputs are returned (an (N, ...)
        array of the model's dtype, e.g. softmax probabilities).
        """

        self.__sync_backend()
        X = asnumpy(X) if is_cupy_array(X) else np.array(X).copy()
        outputs = []
        device_labels = []   # on-device argmax results; one transfer at the end
        # On the device, per-batch looping only adds kernel-launch overhead:
        # one big forward is simpler and faster (the only constraint is
        # device memory, which fits any teaching-scale dataset).  On the
        # host, keep the caller's batch size.
        if on_gpu():
            batch_size = max(batch_size, X.shape[0])
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            batch_X = X[start_idx:end_idx]
            batch_output = self.__forward(batch_X, is_training=False)
            if (self.__idx2label is not None and is_cupy_array(batch_output)
                    and batch_output.ndim == 2):
                # decode labels on the device and defer the transfer: only
                # the small integer labels cross the bus, once, at the end
                # (per-batch asnumpy of the full output matrix would stall
                # the GPU pipeline on every batch)
                device_labels.append(_backend.xp.argmax(batch_output, axis=1))
                continue
            batch_output = asnumpy(batch_output)  # back to the host for label decoding
            if self.__idx2label is not None:
                if _ck is not None and batch_output.ndim == 2 and batch_output.dtype == np.float64:
                    batch_output = np.array([self.__idx2label[i] for i in _ck.argmax_rows(batch_output)])
                else:
                    batch_output = np.array([self.__idx2label[np.argmax(pred, axis=0)] for pred in batch_output])
            elif batch_output.ndim == 2 and batch_output.shape[1] == 1:
                batch_output = batch_output.flatten()
            outputs.append(batch_output)
        if device_labels:
            labels = asnumpy(_backend.xp.concatenate(device_labels))
            outputs.insert(0, np.array([self.__idx2label[i] for i in labels]))
        return np.concatenate(outputs)
    
    def summary(self):
        print("Model: Sequential")
        print("_" * 65)
        print(f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)

        total_params = 0

        for layer_name, layer in self.layers.items():
            params = np.sum([np.prod(v.shape) for v in layer.params.values()]) if hasattr(layer, 'params') else 0
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                output_shape = layer.output_shape
            else:
                output_shape = layer.output_dim if hasattr(layer, 'output_dim') else 'N/A'
            print(f"{layer_name:<20} {str(output_shape):<20} {params:<10,}")

            total_params += params
        
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print("_" * 65)

    def __build(
            self,
        ) -> None:

        output_dim = None
        output_shape = None
        for layer in self.layers.values():
            # 4D-aware layers (Conv2D, MaxPool2D, ...) need the full input
            # shape; the scalar output_dim is not enough for them.
            if output_shape is not None and hasattr(layer, 'set_input_shape'):
                layer.set_input_shape(output_shape)
            if output_dim and hasattr(layer, 'init_params'):
                layer.init_params(output_dim)
            if hasattr(layer, 'set_output_dim'):
                layer.set_output_dim(output_dim)
            output_dim = layer.output_dim
            output_shape = getattr(layer, 'output_shape', None)

        # Initializers always draw in float64; cast the freshly created
        # state when the model asks for another compute dtype.
        if self.__dtype is not None:
            for layer in self.layers.values():
                for attr in ("params", "grads"):
                    d = getattr(layer, attr, None)
                    if isinstance(d, dict):
                        for key, value in d.items():
                            if value.dtype != self.__dtype:
                                d[key] = asarray(value, dtype=self.__dtype)
                for attr in ("moving_mean", "moving_variance"):
                    value = getattr(layer, attr, None)
                    if value is not None and value.dtype != self.__dtype:
                        setattr(layer, attr, asarray(value, dtype=self.__dtype))
    
    def __criterion(
            self,
            y: np.ndarray,
            y_hat: np.ndarray,
        ) -> Tuple[np.float64, np.ndarray]:

        if y.ndim == 1 and y_hat.ndim == 2:
            y = y.reshape(-1, 1)   # view; the loss functions only read y
        y = asarray(y)             # move labels to the same device as y_hat
        if y.dtype != y_hat.dtype:
            # follow the model dtype, or a float64 y would promote a
            # float32 model's loss/gradients back to float64
            y = asarray(y, dtype=y_hat.dtype)
        loss = self.__loss_func(y, y_hat)
        grad = self.__loss_func.grad(y, y_hat)
        if grad.dtype != y_hat.dtype:
            # same promotion leak on the gradient side (loss functions
            # like CCE clip with Python scalars)
            grad = asarray(grad, dtype=y_hat.dtype)
        # Each layer chains through its own activation inside its backward,
        # so the criterion only computes the loss and its gradient w.r.t.
        # the network output.
        return item(loss), grad

    def __forward(
            self,
            inputs,
            is_training: bool = True,
        ) -> np.ndarray:

        inputs = asarray(inputs)   # single entry point: host batches move here
        if self.__dtype is not None and inputs.dtype != self.__dtype:
            inputs = asarray(inputs, dtype=self.__dtype)
        for layer in self.layers.values():
            if not hasattr(layer, 'forward'):
                continue
            output = layer.forward(inputs, is_training)
            # keep the model dtype at every layer boundary: some backend
            # ops (e.g. cupy's clip/maximum) promote float32 arrays to
            # float64 when given Python scalar arguments
            if self.__dtype is not None and output.dtype != self.__dtype:
                output = asarray(output, dtype=self.__dtype)
            inputs = output
        return output

    def __sync_backend(
            self,
        ) -> None:
        """Reconcile model state (params/grads, BatchNorm stats, optimizer
        state) with the active backend, in either direction.

        Idempotent: arrays already on the right device pass through
        unchanged.  Called at the entry of fit/predict/evaluate so models
        built before a set_backend("cupy") switch -- or after switching
        back to numpy -- keep working automatically.
        """
        def _sync(a):
            if on_gpu():
                return a if is_cupy_array(a) else asarray(a)
            return asnumpy(a) if is_cupy_array(a) else a

        for layer in self.layers.values():
            for attr in ("params", "grads"):
                d = getattr(layer, attr, None)
                if isinstance(d, dict):
                    for key, value in d.items():
                        d[key] = _sync(value)
            for attr in ("moving_mean", "moving_variance"):
                value = getattr(layer, attr, None)
                if value is not None:
                    setattr(layer, attr, _sync(value))

        # Optimizer moments/velocities are dicts created lazily on the
        # first update; sync them too, or a backend switch mid-training
        # would mix devices in the update.  Two shapes exist: nested
        # {layer_index: {param: array}} (moments, velocity, ...) and flat
        # {param: array} (SGD's velocity_prev) -- handle both.
        opt = getattr(self, "optimizer", None)
        if opt is not None:
            for attr, value in vars(opt).items():
                if isinstance(value, dict):
                    for key, entry in value.items():
                        if isinstance(entry, dict):
                            for param, arr in entry.items():
                                if is_numpy_array(arr) or is_cupy_array(arr):
                                    entry[param] = _sync(arr)
                        elif is_numpy_array(entry) or is_cupy_array(entry):
                            value[key] = _sync(entry)
    
    def __backward(
            self, 
            grad,
        ) -> None:
        
        reversed_layers = reversed(self.layers.values())
        grad = next(reversed_layers).backward(grad)
        for layer in reversed_layers:
            if not hasattr(layer, 'backward'):
                continue
            grad = layer.backward(grad)
    
    @property
    def layers(self):
        return self.__layers
    
    @property
    def history(self):
        return self.__history
    
    @property
    def parameters(self):
        return {idx: {key: param for key, param in layer.params.items()} for idx, layer in self.layers.items() if hasattr(layer, 'params')}