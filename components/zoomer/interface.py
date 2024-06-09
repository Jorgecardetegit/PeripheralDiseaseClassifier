from abc import ABC, abstractmethod

class Zoom(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def zoom_into_bounding_box(image, bounding_boxes, output_size) -> list:
        pass


