import cv2
import numpy as np 

class adaptativeMask: 

    def __init__(self):
        super().__init__()
        
    def binary_mask(self, grayscale_image) -> np.ndarray:

        """
        This method creates a binary mask with an adaptative threshold, we have to define how to calculate it. 
        """
        threshold_value = np.mean(grayscale_image) # Modify this line to calculate the threshold value. 

        _, binary_mask = cv2.threshold(src=grayscale_image, 
                                        thresh=np.min(grayscale_image + threshold_value),    #np.max
                                        maxval=255, 
                                        type=cv2.THRESH_BINARY) 
        return binary_mask
    

