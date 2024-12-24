from .categorical_crossentropy import CategoricalCrossEntropy
from .mse import MSE

class _LossMapper:
    loss_mapper = {
        'mse': MSE(),
        'categorical_crossentropy': CategoricalCrossEntropy(), 
        'sparse_categorical_crossentropy': CategoricalCrossEntropy(name="sparse_categorical_crossentropy"),
    }
    
    def __getitem__(self, name: str):
        if not name.lower() in self.loss_mapper:
            raise ValueError(f'Loss {name} not found')
        else:
            return self.loss_mapper.get(name)