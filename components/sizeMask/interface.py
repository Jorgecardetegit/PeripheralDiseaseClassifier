from abc import ABC, abstractmethod

class SizeMask(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def size_mask(self, bounding_boxes) -> list:
        pass
