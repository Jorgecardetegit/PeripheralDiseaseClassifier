from mainClass import CellImageProcessor, CellVisualizer, cellExtractor
from transformers import ConvNextForImageClassification

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

import tensorflow as tf

trico_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\TRICOLEUCEMIA"
sezary_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases de datos\FOTOS JORGE\SEZARY"
velloso_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\L. VELLOSOS"
linfoblastica_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\L.A. LINFOBLµSTICA"
normales_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\LINFOCITOS NORMALES"
manto_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\LINFOMA DEL MANTO"
LLC_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\LLC"
plasmatica_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases_de_datos\FOTOS JORGE\CLULA PLASMµTICA"


trico = os.listdir(trico_path)
velloso = os.listdir(velloso_path)
linfoblastica = os.listdir(linfoblastica_path)
normales = os.listdir(normales_path)
manto = os.listdir(manto_path)
LLC = os.listdir(LLC_path)
plasmatica = os.listdir(plasmatica_path)

model_id = "JorgeGIT/finetuned-Leukemia-cell"
model = ConvNextForImageClassification.from_pretrained(model_id)


for image_filename in manto:
    image_path = os.path.join(manto_path, image_filename)
    img = cv2.imread(image_path)

    # Split the image into its channel components
    blue_channel, green_channel, red_channel = cv2.split(img)

    # Calculate the median and maximum values for each channel
    print(np.mean(red_channel))
    print(np.mean(green_channel))
    print(np.mean(blue_channel))

    print(np.min(red_channel))
    print(np.min(green_channel))
    print(np.min(blue_channel))


    processor = CellImageProcessor(image = img, model = model,red_weight=0.3, green_weight=0.59, blue_weight=0.11)

    enhanced_img = processor.enhance_purple_to_grayscale()

    print(np.min(enhanced_img))
    print(np.max(enhanced_img))

    # Get the binary mask of the grayscale image
    binary_img = processor.binary_mask(enhanced_img)

    # Find the object locations in the binary image
    objects = processor.find_objects_locations(binary_img)
    print(objects)

    filtered_objects = processor.size_mask(objects)
    print(filtered_objects)

    classifications = processor.classify_cells(filtered_objects)
    print(classifications)

    visualizer = CellVisualizer(img, binary_img, filtered_objects, classifications)
    
    annotated_img_classes = visualizer.visualize()

    fig = plt.figure(figsize=(10, 10))

    # Add a subplot for the original image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img)
    ax1.title.set_text('Original Image')

    # Add a subplot for the binary image
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(binary_img, cmap='gray')
    ax2.title.set_text('Binary Image')

    # Add a subplot for the classified cells
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.imshow(visualizer.visualize())
    ax4.title.set_text('Classified Cells')

    # Show the figure with the subplots
    plt.tight_layout()
    plt.show()




