from . import functional as F

class Constant:
    def __init__(self, value=0.0):
        self.value = value
    
    def __call__(self, shape):
        return F.constant(shape, self.value)

class GlorotNormal:
    def __call__(self, shape):
        return F.xaiver_normal(shape)

class GlorotUniform:
    def __call__(self, shape):
        return F.xaiver_uniform(shape)
    
class HeNormal:
    def __call__(self, shape):
        return F.kaiming_normal(shape)

class HeUniform:
    def __call__(self, shape):
        return F.kaiming_uniform(shape)

class Ones:
    def __call__(self, shape):
        return F.ones(shape)

class RandomNormal:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, shape):
        return F.normal(shape, self.mean, self.std)

class RandomUniform:
    def __init__(self, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval
    
    def __call__(self, shape):
        return F.uniform(shape, self.minval, self.maxval)

class Zeros:
    def __call__(self, shape):
        return F.zeros(shape)