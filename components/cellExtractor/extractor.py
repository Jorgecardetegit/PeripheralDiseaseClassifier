import cv2 
from components.cellExtractor.interface import Extractor

class cellExtractor(Extractor):
    def __init__(self):
        super().__init__()
        self.name = "cellExtractor"
        self.fixed_size = 300

    def extract_cells(self, image, bounding_boxes, classifications, zoom=True, output_size=(256, 256)):
            extracted_images = []
            
            for (x, y, w, h), classification in zip(bounding_boxes, classifications):
                if zoom:
                    # Calculate the crop size based on the zoom factor; here we assume zoom factor logic is built in
                    crop_size = int(min(w, h) * 0.5)  # for example, zoom to 50% of the smaller dimension
                else:
                    # Use fixed size if not zooming, ensuring it's within the dimensions of the original bounding box
                    crop_size = min(self.fixed_size, w, h)
                
                # Calculate the center and derive the top-left corner from it
                center_x = x + w // 2
                center_y = y + h // 2
                start_x = max(center_x - crop_size // 2, 0)
                start_y = max(center_y - crop_size // 2, 0)
                end_x = min(start_x + crop_size, image.shape[1])
                end_y = min(start_y + crop_size, image.shape[0])

                # Check if the calculated dimensions are valid
                if end_x > start_x and end_y > start_y:
                    cropped_image = image[start_y:end_y, start_x:end_x]
                    
                    if cropped_image.size > 0:
                        # Resize the cropped image to the desired output size
                        resized_image = cv2.resize(cropped_image, output_size)
                        
                        # Append the processed image and its metadata
                        extracted_images.append((resized_image, classification, (start_x, start_y, end_x, end_y, center_x, center_y)))

            return extracted_images