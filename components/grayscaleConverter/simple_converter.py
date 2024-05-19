import cv2
import numpy as np
from components.grayscaleConverter.interface import Converter

class SimpleConverter(Converter):
    def __init__(self):
        super().__init__() 

    def enhance_purple_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image


