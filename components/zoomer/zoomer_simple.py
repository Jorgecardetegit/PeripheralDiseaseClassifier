import cv2
from components.zoomer.interface import Zoom

class Zoomer(Zoom):
    def __init__(self):
        super().__init__()
        self.name = "Zoomer"

    def zoom_into_bounding_box(self, image, bounding_boxes, output_size=(256, 256)):
        zoomed_images = []
        for x, y, w, h in bounding_boxes:
            cx, cy = x + w // 2, y + h // 2  # Center of the bounding box
            crop_size = min(w, h)  # Use the smaller dimension to ensure the crop is within the bbox
            
            # Calculate the crop coordinates, ensuring they are within the image bounds
            start_x = max(cx - crop_size // 2, 0)
            start_y = max(cy - crop_size // 2, 0)
            end_x = min(cx + crop_size // 2, image.shape[1])
            end_y = min(cy + crop_size // 2, image.shape[0])

            # Crop the image
            cropped_img = image[start_y:end_y, start_x:end_x]

            # Resize the cropped image to the desired output size
            resized_img = cv2.resize(cropped_img, output_size)

            # Store the zoomed image
            zoomed_images.append(resized_img)

        return zoomed_images