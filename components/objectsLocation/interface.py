from abc import ABC, abstractmethod
import numpy as np

class objectsLocation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def find_objects_locations(self, binary_image, offset=10) -> list: 
        pass
