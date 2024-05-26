from components.cellExtractor.interface import Extractor

class cellExtractor(Extractor):
    def __init__(self):
        super().__init__()
        self.name = "cellExtractor"
        self.fixed_size = 300

    def extract_cells(self, image, bounding_boxes, classifications):
        extracted_images = []

        for (x, y, w, h), classification in zip(bounding_boxes, classifications):
            center_x = x + w // 2
            center_y = y + h // 2
            start_x = max(center_x - self.fixed_size // 2, 0)
            start_y = max(center_y - self.fixed_size // 2, 0)
            end_x = min(start_x + self.fixed_size, image.shape[1])
            end_y = min(start_y + self.fixed_size, image.shape[0])

            if end_x > start_x and end_y > start_y:  # Ensure the cropped area is valid
                cropped_image = image[start_y:end_y, start_x:end_x]
                if cropped_image.size > 0:
                    # Append the image, classification, and coordinates
                    extracted_images.append((cropped_image, classification, (start_x, start_y, end_x, end_y, center_x, center_y)))

        return extracted_images