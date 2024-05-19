import cv2 
import numpy as np

from components.grayscaleConverter.interface import Converter

class CustomConverter(Converter):

  def __init__(self):
      super().__init__() 

  def enhance_purple_to_grayscale(self, image: np.ndarray) -> np.ndarray:
      # Example custom conversion: Average the R, G, B channels with custom weights
      # Custom weights for R, G, B channels (modify as needed)
      r_weight = 0.3
      g_weight = 0.59
      b_weight = 0.11

      # Split the image into its R, G, B channels
      b, g, r = cv2.split(self.image)

      # Compute the custom grayscale image using the specified weights
      custom_gray_image = (r * r_weight + g * g_weight + b * b_weight).astype(np.uint8)

      return custom_gray_image



