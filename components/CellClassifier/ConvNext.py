import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from typing import Tuple, List

from components.CellClassifier.interface import Classifier

from transformers import ConvNextForImageClassification

class ConvNext(Classifier): 
    def __init__(self):
        super().__init__()
        self.name = "ConvNext"
        self.model_id = "JorgeGIT/finetuned-Leukemia-cell"
        self.model = ConvNextForImageClassification.from_pretrained(self.model_id)
        self.model.eval()

    def classify_cells(self, image, filtered_bounding_boxes) -> list:
        image = Image.fromarray(image)
        classifications = []
        probs = []

        for bounding_box in filtered_bounding_boxes:
            # Apply bounding box crop here if necessary. This code assumes the full image is processed.
            preprocess = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)

            predicted_class_index = outputs.logits.argmax(-1).item()
            predicted_class_name = self.class_names.get(predicted_class_index + 1, "Unknown")  # +1 if class indices are 1-based
            classifications.append(predicted_class_name)

            probabilities = torch.softmax(outputs.logits, dim=-1)
            # Convert probabilities to a list of tuples (class_name, probability)
            class_probabilities = [(self.class_names.get(idx + 1, "Unknown"), prob.item()) for idx, prob in enumerate(probabilities.squeeze())]
            probs.append(class_probabilities)

        return classifications