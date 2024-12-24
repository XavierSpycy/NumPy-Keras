import math
from typing import (
    List,
    Callable,
)

class LRScheduler:
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def on_epoch_end(self, *args, **kwargs):
        return self.step(*args, **kwargs)

class MultiplicativeLR(LRScheduler):
    def __init__(
            self, 
            lr_lambda: Callable[[int], float],
        ) -> None:

        """
        Initialize the MultiplicativeLR.

        Parameters:
        - lr_lambda (function): A function that takes the current epoch and returns the factor by which to multiply the learning rate.
        """

        self.lr_lambda = lr_lambda
        self.current_epoch = 0
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each epoch. This will update the learning rate if necessary.
        """

        self.current_epoch += 1
        model.optimizer.learning_rate *= self.lr_lambda(self.current_epoch)

class StepLR(LRScheduler):
    def __init__(
            self, 
            step_size: int, 
            gamma: float = 0.1,
        ) -> None:

        """
        Initialize the StepLR.
        
        Parameters:
        - step_size (int): The number of epochs after which to decay the learning rate.
        - gamma (float, optional): The factor by which to multiply the learning rate when decaying.
        """

        self.step_size = step_size
        self.gamma = gamma
        self.current_iters = 0
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs
        ) -> None:

        """
        Call this method after each epoch. This will update the learning rate if necessary.
        """

        self.current_iters += 1
        if self.current_iters % self.step_size == 0:
            model.optimizer.learning_rate *= self.gamma

class MultiStepLR(LRScheduler):
    def __init__(
            self, 
            milestones: List[int], 
            gamma: float = 0.1,
        ) -> None:

        """
        Initialize the MultiStepLR.

        Parameters:
        - milestones (list): List of epoch indices. Must be increasing.
        - gamma (float, optional): The factor by which to multiply the learning rate when decaying.
        """

        self.milestones = iter(milestones)
        self.current_milestone = next(self.milestones, float('inf'))
        self.gamma = gamma
        self.current_iters = 0
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        self.current_iters += 1
        if self.current_iters == self.current_milestone:
            model.optimizer.learning_rate *= self.gamma
            self.current_milestone = next(self.milestones, float('inf'))

class ConstantLR(LRScheduler):
    def __init__(
            self, 
            factor: float = 1./3, 
            total_iters: int = 5,
        ) -> None:

        """
        Initialize the ConstantLR.

        Parameters:
        - factor (float, optional): The factor by which to multiply the learning rate when decaying.
        - total_iters (int, optional): The number of iterations after which to decay the learning rate.
        """

        self.factor = factor
        self.total_iters = total_iters
        self.init_lr = None
        self.constant_lr = self.init_lr * self.factor
        self.current_iters = 0

    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        if self.init_lr is None:
            self.init_lr = model.optimizer.learning_rate
        self.current_iters += 1
        if self.current_iters < self.total_iters:
            model.optimizer.learning_rate = self.constant_lr
        elif self.current_iters == self.total_iters:
            model.optimizer.learning_rate = self.init_lr

class LinearLR(LRScheduler):
    def __init__(
            self, 
            start_factor: float = 1./3,
            end_factor: float = 1./10,
            total_iters: int = 5,
        ) -> None:
        
        """
        Initialize the LinearLR.

        Parameters:
        - start_factor (float, optional): The factor by which to multiply the learning rate when decaying.
        - end_factor (float, optional): The factor by which to multiply the learning rate when decaying.
        - total_iters (int, optional): The number of iterations after which to decay the learning rate.
        """

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        if self.current_iters < self.total_iters:
            model.optimizer.learning_rate += self.linear_update
        else:
            model.optimizer.learning_rate *= self.end_factor
    
    def on_train_begin(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method at the start of each epoch. This will reset the learning rate to the initial value.
        """

        model.optimizer.learning_rate *= self.start_factor
        self.init_lr = model.optimizer.learning_rate
        self.linear_update = (self.init_lr - self.init_lr * self.end_factor) / self.total_iters

class ExponentialLR(LRScheduler):
    def __init__(
            self, 
            gamma: float,
        ) -> None:

        """
        Initialize the ExponentialLR.

        Parameters:
        - gamma (float, optional): The factor by which to multiply the learning rate when decaying.
        """

        self.gamma = gamma
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        model.optimizer.learning_rate *= self.gamma

class PolynomialLR(LRScheduler):
    def __init__(
            self, 
            total_iters: int, 
            power: float,
        ) -> None:

        """
        Initialize the PolynomialLR.

        Parameters:
        - max_iters (int): The number of iterations after which to decay the learning rate.
        - power (float): The power to which to raise the iteration number.
        """

        self.total_iters = total_iters
        self.power = power
        self.current_iters = 0

    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        self.current_iters += 1
        model.optimizer.learning_rate *= (1 - self.current_iters / self.total_iters) ** self.power

class CosineAnnealingLR(LRScheduler):
    def __init__(
            self, 
            T_max: int, 
            eta_min: float = 0,
        ) -> None:

        """
        Initialize the CosineAnnealingLR.

        Parameters:
        - T_max (int): The maximum number of iterations in the cycle.
        - eta_min (float, optional): The minimum learning rate.
        """

        self.T_max = T_max
        self.eta_min = eta_min
        self.T_cur = 0
    
    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        self.T_cur += 1
        model.optimizer.learning_rate = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_max))
    
    def on_train_begin(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method at the start of each epoch. This will reset the learning rate to the initial value.
        """

        self.eta_max = model.optimizer.learning_rate

class ReduceLROnPlateau(LRScheduler):
    def __init__(
            self,
            monitor: str = 'val_loss',
            mode: str = 'min',
            factor: float = 0.1,
            patience: int = 10,
            threshold: float = 1e-4,
            threshold_mode: str = 'rel',
            cooldown: int = 0,
            min_lr: float = 0,
            eps: float = 1e-8,
        ) -> None:

        """
        Initialize the ReduceLROnPlateau.

        Parameters:
        - mode (str, optional): One of {'min', 'max'}. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing.
        - factor (float, optional): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        - patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced.
        - threshold (float, optional): Threshold for measuring the new optimum, to only focus on significant changes.
        - threshold_mode (str, optional): One of {'rel', 'abs'}. In 'rel' mode, dynamic_threshold = best * (1 + threshold) in 'abs' mode, dynamic_threshold = best + threshold.
        - cooldown (int, optional): Number of epochs to wait before resuming normal operation after lr has been reduced.
        - min_lr (float, optional): A lower bound on the learning rate.
        - eps (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.
        """

        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.wait = 0
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.cooldown_counter = cooldown

    def step(
            self, 
            model, 
            *args, 
            **kwargs,
        ) -> None:

        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """

        curr = model.history.metrics[self.monitor][-1]

        if self.mode == 'min':
            if curr < self.best * (1 - self.threshold if self.threshold_mode == 'rel' else self.threshold):
                self.best = curr
                self.wait = 0
        else:
            if curr > self.best * (1 + self.threshold if self.threshold_mode == 'rel' else self.threshold):
                self.best = curr
                self.wait = 0

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.wait >= self.patience:
                new_lr = max(self.min_lr, model.optimizer.learning_rate * self.factor)
                if model.optimizer.learning_rate - new_lr > self.eps:
                    model.optimizer.learning_rate = new_lr
                self.cooldown_counter = self.cooldown
                self.wait = 0
            else:
                self.wait += 1