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

class Sequential:
    __idx2label = None

    def __init__(
            self, 
            layers: List[
                Union[
                    layers.Activation, 
                    layers.BatchNormalization,
                    layers.Dense,
                    layers.Dropout,
                    layers.Flatten,
                    layers.Input,
                ]
            ] = [],
        ) -> None:

        self.__layer_counter = defaultdict(int)
        self.__layers = {}
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

        X = np.array(X).copy()
        y = np.array(y).copy()

        y_pred = self.predict(X, batch_size)

        if self.__metrics:
            return metrics._MetricMapper()[self.__metrics[0]](y, y_pred)
        else:
            return self.__loss_func(y, y_pred)
    
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

        X_train = np.array(X)
        y_train = np.array(y)
        X_test = None
        y_test = None

        self.stop_training = False
        self.__best_weights = None
        
        if self.__loss_func.name == 'sparse_categorical_crossentropy':
            y_train, self.__idx2label = utils.one_hot_encode(y_train)
        
        if validation_data is not None:
            X_test, y_test = validation_data
            if self.__loss_func.name == 'sparse_categorical_crossentropy':
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
                y_pred_train = self.predict(
                    X_train, 
                    batch_size=validation_batch_size if validation_batch_size is not None else batch_size)
                if X_test is not None:
                    y_pred_test = self.predict(X_test, batch_size=validation_batch_size if validation_batch_size is not None else batch_size)
                else:
                    y_pred_test = None
                
                if y_pred_test is not None:
                    if not "val_loss" in self.__history.metrics:
                        self.__history.metrics["val_loss"] = []
                    self.__history.metrics["val_loss"].append(
                        self.__loss_func(y_test, self.__forward(X_test, is_training=False))
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

                    if hasattr(callback, 'save_best_weight') and callback.save_best_weight:
                        self.__best_weights = self.parameters # {idx: {key: param for key, param in layer.params.items()} for idx, layer in self.layers.items() if hasattr(layer, 'params')}
            
            if self.stop_training:
                break
        
        if self.__best_weights is not None:
                for idx, layer in self.parameters:
                    for param in layer.params:
                        self.__layers[idx].params[param] = self.__best_weights[idx][param]
        
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
        
        X = np.array(X).copy()
        output = np.zeros(X.shape[0])
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            batch_X = X[start_idx:end_idx]
            batch_output = self.__forward(batch_X, is_training=False)
            if batch_output.ndim == 2 and batch_output.shape[1] == 1:
                batch_output = batch_output.flatten()
            if self.__idx2label is not None:
                batch_output = np.array([self.__idx2label[np.argmax(pred, axis=0)] for pred in batch_output])
            output[start_idx:end_idx] = batch_output
        return output
    
    def summary(self):
        print("Model: Sequential")
        print("_" * 65)
        print(f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)

        total_params = 0

        for layer_name, layer in self.layers.items():
            params = np.sum([np.prod(v.shape) for v in layer.params.values()]) if hasattr(layer, 'params') else 0
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
        prev_layer_activation = None
        prev_layer_activation_config = {}
        for layer in self.layers.values():
            if output_dim and hasattr(layer, 'init_params'):
                layer.init_params(output_dim)
            if hasattr(layer, 'set_activation_deriv'):
                layer.set_activation_deriv(prev_layer_activation, prev_layer_activation_config)
            if hasattr(layer, 'set_output_dim'):
                layer.set_output_dim(output_dim)
            prev_layer_activation = layer.activation if hasattr(layer, 'activation') else prev_layer_activation
            prev_layer_activation_config = layer.activation_config if hasattr(layer, 'activation_config') else prev_layer_activation_config
            output_dim = layer.output_dim
    
    def __criterion(
            self, 
            y: np.ndarray, 
            y_hat: np.ndarray,
        ) -> Tuple[np.float64, np.ndarray]:

        y_ = y.copy()
        if y.ndim == 1 and y_hat.ndim == 2:
            y_ = y_.reshape(-1, 1)
        activation_deriv = list(self.layers.values())[-1].activation_deriv
        loss = self.__loss_func(y_, y_hat)
        grad = self.__loss_func.grad(y_, y_hat) * activation_deriv(y_hat)
        return loss, grad

    def __forward(
            self, 
            inputs,
            is_training: bool = True,
        ) -> np.ndarray:
        
        for layer in self.layers.values():
            if not hasattr(layer, 'forward'):
                continue
            output = layer.forward(inputs, is_training)
            inputs = output
        return output
    
    def __backward(
            self, 
            grad,
        ) -> None:
        
        grad = list(self.layers.values())[-1].backward(grad)
        for layer in reversed(list(self.layers.values())[:-1]):
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