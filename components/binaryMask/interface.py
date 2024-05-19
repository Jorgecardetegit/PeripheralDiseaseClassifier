from abc import ABC, abstractmethod
import numpy as np

class BinaryMasker(ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def binaryMask(self,
                   grayscale_image
                   ) -> np.ndarray:
        pass


