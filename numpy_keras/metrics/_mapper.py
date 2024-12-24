from . import functional as F

class _MetricMapper:
    metric_mapper = {
        'accuracy': F.accuracy,
        'mean_squared_error': F.mean_squared_error,
        'r2_score': F.r2_score,
    }
    
    def __getitem__(self, name: str):
        if name not in self.metric_mapper:
            raise ValueError(f'Unknown metric: {name}')
        return self.metric_mapper[name]