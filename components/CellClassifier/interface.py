from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Classifier(ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def classify_cells(self) -> List[Tuple[np.ndarray, str, Tuple[int, int, int, int, int, int]]]:
        pass