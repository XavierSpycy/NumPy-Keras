from ._base import Optimizer
from .adadelta import Adadelta
from .adagrad import Adagrad
from .sgd import SGD
from .adam import Adam

class _OptimMapper:
    optimizer_mapper = {
        'sgd': SGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
    }
    
    def __getitem__(self, name: str) -> Optimizer:
        if not name.lower() in self.optimizer_mapper:
            raise ValueError(f'Optimizer {name} not found')
        elif name.lower() == 'sgd':
            return self.optimizer_mapper.get(name)(learning_rate=0.1)
        else:
            return self.optimizer_mapper.get(name)()