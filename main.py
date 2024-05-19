from mainClass import CellImageProcessor, CellVisualizer, cellExtractor
from transformers import ConvNextForImageClassification

import cv2
import numpy as np

model_id = "JorgeGIT/finetuned-Leukemia-cell"
model = ConvNextForImageClassification.from_pretrained(model_id)

def process_image(image) -> tuple:
    processor = CellImageProcessor(image=image, model=model)

    enhanced_img = processor.enhance_purple_to_grayscale()
    binary_img = processor.binary_mask(enhanced_img)
    objects = processor.find_objects_locations(binary_img)
    filtered_objects = processor.size_mask(objects)
    classifications, probabilities = processor.classify_cells(filtered_objects)

    return binary_img, filtered_objects, classifications, probabilities

def visualize_results(image,binary_img, filtered_objects, classifications) -> tuple:
    visualizer = CellVisualizer(image, binary_img, filtered_objects, classifications)

    annotated_img = visualizer.visualize()

    mask = visualizer.visualize_mask()

    bounding_boxes = visualizer.visualize_bounding_boxes()

    classified_img = visualizer.visualize_classifications()

    return annotated_img, mask, bounding_boxes, classified_img

def extract_cells(image, filtered_objects, classifications) -> list:
    extractor = cellExtractor(image, filtered_objects, classifications)

    extracted_cells = extractor.extract_squares()

    return extracted_cells

def load_image_from_file(file_stream):
    data = np.frombuffer(file_stream.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img




