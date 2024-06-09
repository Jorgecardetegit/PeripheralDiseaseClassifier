import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks, argrelextrema

from components.binaryMask.interface import BinaryMasker

class pixelMask(BinaryMasker): 
    def __init__(self):
        super().__init__()

    def find_significant_maxima_and_minima(self, grayscale_image):

        pixel_values = np.array(grayscale_image).flatten()

        counts, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))

        peaks, properties = find_peaks(counts, height=1000, prominence=10000)  

        minima_indices = argrelextrema(counts, np.less)[0]

        # Extract frequencies of maxima and minima
        maxima_frequencies = counts[peaks]
        minima_frequencies = counts[minima_indices]

        print(f"minima frequencies: {minima_frequencies}")

        return peaks, minima_indices, maxima_frequencies, minima_frequencies, counts, bin_edges

    
    def binary_mask(self, grayscale_image) -> np.ndarray:
        maxima_dict = {}
        minima_dict = {}

        maxima_indices, minima_indices, maxima_frequencies, minima_frequencies, counts, bin_edges = self.find_significant_maxima_and_minima(grayscale_image)

        for idx, edge in zip(maxima_indices, bin_edges[maxima_indices]):
            maxima_dict[edge] = counts[idx]
        
        for idx, edge in zip(minima_indices, minima_frequencies):
            minima_dict[idx] = edge
        
        # Find the first maximum index greater than 100
        threshold_value_max = next((bin_edges[idx] for idx in maxima_indices if idx > 100), None)

        # Find a suitable threshold for binarization

        for line in range(len(minima_indices)):
            if minima_indices[line] > threshold_value_max:
                if minima_dict[minima_indices[line - 1]] > maxima_dict[threshold_value_max]/3:
                    print("YES")
                    if minima_dict[minima_indices[line - 2]] > maxima_dict[threshold_value_max]/3:
                        print("YES")
                        threshold_value = minima_indices[line - 4]
                    else:
                        threshold_value = minima_indices[line - 3]
                break
            
            threshold_value = minima_indices[line]
            
        if threshold_value is None:
            threshold_value = np.mean(grayscale_image/2)  # Default threshold if no suitable minima found

        print("Maxima Dict:", maxima_dict)
        print("Minima Dict:", minima_dict)
        print("Selected Threshold Value:", threshold_value)
                        
        _, binary_mask = cv2.threshold(src=grayscale_image, 
                                    thresh=threshold_value, 
                                    maxval=255, 
                                    type=cv2.THRESH_BINARY) 
        return binary_mask

