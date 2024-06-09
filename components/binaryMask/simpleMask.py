import cv2
import numpy as np 

from components.binaryMask.interface import BinaryMasker

class simpleMask(BinaryMasker):
    def __init__(self):
        super().__init__()

    def binary_mask(self, 
                    grayscale_image: np.ndarray, 
                    threshold_value: int) -> np.ndarray:
        
        _, binary_mask = cv2.threshold(src=grayscale_image, 
                                        thresh=np.min(grayscale_image) + threshold_value,    
                                        maxval=255, 
                                        type=cv2.THRESH_BINARY) 
        return binary_mask


    


