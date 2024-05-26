import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

class CellImageProcessor:
    def __init__(self, model, image, red_weight=0.05, green_weight=0.05, blue_weight=0.900,
                 enhancement_factor=1.5, purple_threshold=200, threshold_value = 50):
        self.model = model
        self.image = image
        self.red_weight = red_weight
        self.green_weight = green_weight
        self.blue_weight = blue_weight
        self.enhancement_factor = enhancement_factor
        self.purple_threshold = purple_threshold
        self.threshold_value = threshold_value

    def enhance_purple_to_grayscale(self) -> np.ndarray:
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def binary_mask(self, grayscale_image) -> np.ndarray:
        _, binary_mask = cv2.threshold(src=grayscale_image, 
                                       thresh=np.min(grayscale_image) + self.threshold_value,    #np.max
                                       maxval=255, 
                                       type=cv2.THRESH_BINARY) 
        return binary_mask

    def find_objects_locations(self, binary_image, offset=10) -> list:
        bounding_boxes = []
        padded_image = cv2.copyMakeBorder(src=binary_image, 
                                          top=offset, bottom=offset, left=offset, right=offset, 
                                          borderType=cv2.BORDER_CONSTANT, 
                                          value=0)
        
        contours, _ = cv2.findContours(image=padded_image, 
                                       mode=cv2.RETR_TREE, 
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x - offset, y - offset, w, h))

        if bounding_boxes:
            bounding_boxes = sorted(bounding_boxes, key=lambda box: box[2]*box[3])
            bounding_boxes.pop()
        else:
            return "No bounding boxes found!"
        return bounding_boxes

    def size_mask(self, bounding_boxes) -> list:
        filtered_bounding_boxes = []
        
        for x, y, w, h in bounding_boxes:
            if (w*h >= bounding_boxes[-1][2]*bounding_boxes[-1][3] / 2):
                filtered_bounding_boxes.append((x, y, w, h))

        return filtered_bounding_boxes

    def classify_cells(self, filtered_bounding_boxes) -> list:
        image = Image.fromarray(self.image)
        classifications = []
        probs = []
        
        for bounding_box in filtered_bounding_boxes:
            preprocess = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),])
            image_tensor = preprocess(image).unsqueeze(0) 
            

            with torch.no_grad():
                outputs = self.model(image_tensor)

            predicted_class = outputs.logits.argmax(-1).item()
            classifications.append(predicted_class)

            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Convert probabilities to a list of tuples (class_index, probability)
            class_probabilities = [(idx, prob.item()) for idx, prob in enumerate(probabilities.squeeze())]

            probs.append(class_probabilities)

        return classifications, probs
    
    def global_probs(self, probs):
        global_probs = []
        for prob in probs:
            global_probs.append(prob[0][1])
        return global_probs

class CellVisualizer():
    def __init__(self, image, binary_mask, bounding_boxes, classifications, box_thickness=4, fixed_size=500):
        self.image = image
        self.bounding_boxes = bounding_boxes
        self.classifications = classifications
        self.box_thickness = box_thickness
        self.fixed_size = fixed_size
        self.binary_mask = binary_mask

    def visualize(self):
        image_with_boxes = self.image.copy()

        for (x, y, w, h), classification in zip(self.bounding_boxes, self.classifications):
            # Calculate the center of the original bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Adjust starting point to center the new fixed-size box around the original center
            start_x = center_x - self.fixed_size // 2
            start_y = center_y - self.fixed_size // 2
            
            cv2.rectangle(image_with_boxes, (start_x, start_y), 
                          (start_x + self.fixed_size, start_y + self.fixed_size), 
                          (0, 255, 0), self.box_thickness)

            # Display the classification label above the centered bounding box
            cv2.putText(image_with_boxes, f"Class: {classification}", 
                        (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 128), self.box_thickness)
        
        return image_with_boxes
    
    def visualize_mask(self):
        image_with_mask = self.image.copy()

        image_with_mask[self.binary_mask == 0] = 0

        return image_with_mask

    def visualize_bounding_boxes(self):
        image_with_only_boxes = self.image.copy()

        for (x, y, w, h), classification in zip(self.bounding_boxes, self.classifications):
            cv2.rectangle(image_with_only_boxes, (x, y), (x + w, y + h), (0, 255, 0), self.box_thickness)

        return image_with_only_boxes
    
    def visualize_classifications(self):
        image_with_only_classifications = self.image.copy()

        for (x, y, w, h), classification in zip(self.bounding_boxes, self.classifications):
            cv2.putText(image_with_only_classifications, f"Class: {classification}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 128), self.box_thickness)

        return image_with_only_classifications

class cellExtractor():
    def __init__(self, image, bounding_boxes, classifications, fixed_size=300):
        self.image = image
        self.bounding_boxes = bounding_boxes
        self.classifications = classifications
        self.fixed_size = fixed_size

    def extract_squares(self):
        extracted_images = []

        for (x, y, w, h), classification in zip(self.bounding_boxes, self.classifications):
            center_x = x + w // 2
            center_y = y + h // 2
            start_x = max(center_x - self.fixed_size // 2, 0)
            start_y = max(center_y - self.fixed_size // 2, 0)
            end_x = min(start_x + self.fixed_size, self.image.shape[1])
            end_y = min(start_y + self.fixed_size, self.image.shape[0])

            if end_x > start_x and end_y > start_y:  # Ensure the cropped area is valid
                cropped_image = self.image[start_y:end_y, start_x:end_x]
                if cropped_image.size > 0:
                    # Append the image, classification, and coordinates
                    extracted_images.append((cropped_image, classification, (start_x, start_y, end_x, end_y, center_x, center_y)))

        return extracted_images
    

