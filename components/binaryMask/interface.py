from abc import ABC, abstractmethod
import numpy as np

class BinaryMasker(ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def binary_mask(self,
                   grayscale_image) -> np.ndarray:
        pass


