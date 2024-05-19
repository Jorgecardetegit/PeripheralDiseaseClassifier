import cv2
from mainClass import CellImageProcessor, CellVisualizer, cellExtractor

import matplotlib.pyplot as plt
from transformers import ConvNextForImageClassification

img_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\L. VELLOSOS\0001-7.JPG"
img = plt.imread(img_path)
plt.imshow(img)
plt.show()

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


binary_img, filtered_objects, classifications, probabilities = process_image(img)    

print(probabilities)

image, mask, filtered_obj, classific = visualize_results(img, binary_img, filtered_objects, classifications)

plt.imshow(mask)
plt.axis('off')
plt.show()

plt.imshow(filtered_obj)
plt.axis('off')
plt.show()

plt.imshow(classific)
plt.axis('off')
plt.show()



fig = plt.figure(figsize=(10, 10))

# Add a subplot for the original image
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(image)

# Add a subplot for the mask
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(mask, cmap='gray')

# Add a subplot for the bounding boxes
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(filtered_obj)

# Add a subplot for the classification image
ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(classific)

plt.show()


