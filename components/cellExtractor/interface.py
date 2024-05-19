from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Extractor(ABC):
    @abstractmethod
    def extract_cells(self) -> List[Tuple[np.ndarray, str, Tuple[int, int, int, int, int, int]]]:
        pass