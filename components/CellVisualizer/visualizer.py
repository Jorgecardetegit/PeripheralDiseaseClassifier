import cv2
from components.CellVisualizer.interface import Visualizer

class CellVisualizer(Visualizer):
    def __init__(self):
        super().__init__()
        self.name = "cellVisualizer"

    def visualize(self, image, bounding_boxes, classifications, box_thickness=4, fixed_size=500):
        image_with_boxes = image.copy()

        for (x, y, w, h), classification in zip(bounding_boxes, classifications):
            # Calculate the center of the original bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Adjust starting point to center the new fixed-size box around the original center
            start_x = center_x - fixed_size // 2
            start_y = center_y - fixed_size // 2
            
            cv2.rectangle(image_with_boxes, (start_x, start_y), 
                          (start_x +fixed_size, start_y + fixed_size), 
                          (0, 255, 0), box_thickness)

            # Display the classification label above the centered bounding box
            cv2.putText(image_with_boxes, f"Class: {classification}", 
                        (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 128), box_thickness)
        
        return image_with_boxes
    
    def visualize_mask(self, image, binary_mask):
        image_with_mask = image.copy()

        image_with_mask[binary_mask == 0] = 0

        return image_with_mask

    def visualize_bounding_boxes(self, image, bounding_boxes, classifications, box_thickness=4, fixed_size=500):
        image_with_only_boxes = image.copy()

        for (x, y, w, h), classification in zip(bounding_boxes, classifications):
            cv2.rectangle(image_with_only_boxes, (x, y), (x + w, y + h), (0, 255, 0), box_thickness)

        return image_with_only_boxes
    
    def visualize_classifications(self, image, bounding_boxes, classifications, box_thickness=4, fixed_size=500):
        image_with_only_classifications = image.copy()

        for (x, y, w, h), classification in zip(bounding_boxes, classifications):
            cv2.putText(image_with_only_classifications, f"Class: {classification}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 128), box_thickness)

        return image_with_only_classifications