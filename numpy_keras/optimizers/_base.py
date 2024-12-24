class Optimizer:
    def __init__(self, learning_rate, *args, **kwargs) -> None:
        self.learning_rate = learning_rate
    
    def update(self, *args, **kwargs):
        raise NotImplementedError