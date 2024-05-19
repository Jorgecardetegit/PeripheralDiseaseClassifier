from abc import ABC, abstractmethod
import numpy as np

class Converter(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def enhance_purple_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        pass