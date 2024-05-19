from skimage import io, color, filters, morphology, measure
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

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


for image_filename in trico:
    image_path = os.path.join(trico_path, image_filename)
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = color.rgb2hsv(image)

    # Isolate the hue, saturation, and value channels
    hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    # Define thresholds for isolating the purple color typically associated with lymphocytes
    hue_thresholds = (0.7, 0.85)  # Purple hue range in HSV
    sat_thresholds = (0.4, 1)     # High saturation for vivid color
    val_thresholds = (0.2, 0.8)   # Mid to high value to avoid very dark/bright artifacts

    # Apply the thresholds to create a binary mask
    hue_mask = (hue > hue_thresholds[0]) & (hue < hue_thresholds[1])
    sat_mask = (sat > sat_thresholds[0]) & (sat < sat_thresholds[1])
    val_mask = (val > val_thresholds[0]) & (val < val_thresholds[1])

    # Combine the masks
    combined_mask = hue_mask & sat_mask & val_mask

    # Apply morphological opening to remove small objects and noise
    cleaned_mask = morphology.opening(combined_mask, morphology.disk(3))

    # Label the objects in the mask
    labeled_mask, num_labels = measure.label(cleaned_mask, return_num=True, connectivity=2)
    properties = measure.regionprops(labeled_mask)

    # Create an image to display the labeled regions overlaid on the original
    labeled_display = color.label2rgb(labeled_mask, image, bg_label=0, alpha=0.4, kind='overlay')

    # Apply more robust morphological operations
    cleaned_mask = morphology.closing(combined_mask, morphology.disk(5))  # Increase the disk size if necessary
    cleaned_mask = morphology.opening(cleaned_mask, morphology.disk(3))

    # Label the objects in the mask
    labeled_mask, num_labels = measure.label(cleaned_mask, return_num=True, connectivity=2)
    properties = measure.regionprops(labeled_mask)

    # Filter out small objects
    min_area = 50  # Set a minimum area threshold
    for region in properties:
        if region.area < min_area:
            labeled_mask[labeled_mask == region.label] = 0

    # Recalculate the number of objects
    labeled_mask, num_labels = measure.label(labeled_mask > 0, return_num=True, connectivity=2)
    labeled_display = color.label2rgb(labeled_mask, image, bg_label=0, alpha=0.4, kind='overlay')

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(cleaned_mask, cmap='gray')
    ax[1].set_title('Refined Mask')
    ax[1].axis('off')

    ax[2].imshow(labeled_display)
    ax[2].set_title(f'Labeled Regions (Objects: {num_labels})')
    ax[2].axis('off')

    plt.show()
