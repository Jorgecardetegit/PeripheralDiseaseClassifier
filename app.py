from flask import Flask, request, render_template, send_file, url_for, make_response, jsonify
import io
import cv2
import numpy as np

import base64
from peripheralManager import PeripheralManager

from backend import save_image_data

from datetime import datetime

def load_image_from_file(file_stream):
    data = np.frombuffer(file_stream.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

peripheral_manager = PeripheralManager()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)  # Reset file pointer after save

            # Read the image for processing and preview
            img = load_image_from_file(in_memory_file)

            grayscale_image, binary_mask, filtered_locations, classifications = peripheral_manager.main_processor(img)
            everything_image, binary_mask, boxes, img_classifications = peripheral_manager.cell_visualizer(img, binary_mask, filtered_locations, classifications, "visualizer")
            cells_extracted = peripheral_manager.cellExtractor(img, filtered_locations, classifications, "extractor")

            # Encode the original image for preview
            in_memory_file.seek(0)
            original_img_encoded = base64.b64encode(in_memory_file.read()).decode('utf-8')

            # Encode the processed image
            _, buffer = cv2.imencode('.jpg', everything_image)
            processed_img_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', binary_mask)
            binary_mask_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', boxes)
            boxes_image_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', img_classifications)
            classification_image_encoded = base64.b64encode(buffer).decode('utf-8')
            
            images_encoded = []

            for cropped_image, classifications, (start_x, start_y, end_x, end_y, center_x, center_y) in cells_extracted:
                _, buffer = cv2.imencode('.jpg', cropped_image)
                img_encoded = base64.b64encode(buffer).decode('utf-8')
                # Save the image and its metadata to the database
                save_image_data(base64.b64decode(img_encoded), classifications, start_x, start_y, end_x, end_y, center_x, center_y, '0.40', datetime.now().strftime('%Y-%m-%d'))

                images_encoded.append(img_encoded)

            return jsonify({
                'original_image': original_img_encoded,
                'processed_image': processed_img_encoded,
                'mask': binary_mask_encoded,
                'square_image': boxes_image_encoded,
                'classification_image': classification_image_encoded,
                'classifications': classifications,
                'extracted_cells': images_encoded
            })

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)