from dataclasses import (
    dataclass,
    field,
)
from typing import (
    List,
    Dict,
)

@dataclass
class History:
    loss: List [float] = field(default_factory=lambda: [])
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {})
    validation_epochs: List[int] = field(default_factory=lambda: [])
    
    def __getitem__(self, key: str):
        return getattr(self, key)