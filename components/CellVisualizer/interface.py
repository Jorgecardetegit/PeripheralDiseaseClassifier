from abc import ABC, abstractmethod
import numpy as np

class Visualizer(ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def visualize(self) -> np.ndarray:
        pass

    @abstractmethod
    def visualize_mask(self) -> np.ndarray:
        pass

    @abstractmethod
    def visualize_bounding_boxes(self) -> np.ndarray:
        pass

    @abstractmethod
    def visualize_classifications(self) -> np.ndarray:
        pass