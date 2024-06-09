import cv2
import numpy as np 
import matplotlib.pyplot as plt 

from components.binaryMask.interface import BinaryMasker

class adaptativeMask(BinaryMasker): 
    def __init__(self):
        super().__init__()
        
    def binary_mask(self, 
                    grayscale_image) -> np.ndarray:

        """
        This method creates a binary mask with an adaptative threshold, we have to define how to calculate it. 
        """
        threshold_value = np.mean(grayscale_image) # Modify this line to calculate the threshold value. 
        print(f"Mean threshold_value: {threshold_value}")
        print(f"Threshold value divided by 5: {np.min(grayscale_image) + threshold_value/5}")
        print(f"Max threshold value: {np.max(grayscale_image)}")
        print(f"Min threshold value: {np.min(grayscale_image)}")

        flattened_image = grayscale_image.flatten()

        # Sort the flattened array to order the pixel values
        sorted_pixels = np.sort(flattened_image)

        first_twenty_pixels = sorted_pixels[:20]

        print(f"First 5 minimum pixel values: {first_twenty_pixels}")

        threshold_value = np.mean(first_twenty_pixels)

        print(f"threshold_value is {threshold_value + threshold_value/1.5}")

        _, binary_mask = cv2.threshold(src=grayscale_image, 
                                        thresh=threshold_value + threshold_value/1.5,    
                                        maxval=255, 
                                        type=cv2.THRESH_BINARY) 
        return binary_mask
    
    

