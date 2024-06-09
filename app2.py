from flask import Flask, request, render_template, send_file, url_for, make_response, jsonify
import io
import cv2
import base64
from main import process_image, visualize_results, load_image_from_file, extract_cells
from backend import save_image_data

from datetime import datetime


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

            binary_image, filtered_objects, classifications, probabilities = process_image(img)
            annotated_img_classes, mask, square_image, classification_image = visualize_results(img, binary_image, filtered_objects, classifications)
            extracted_cells = extract_cells(img, filtered_objects, classifications)

            # Encode the original image for preview
            in_memory_file.seek(0)
            original_img_encoded = base64.b64encode(in_memory_file.read()).decode('utf-8')

            # Encode the processed image
            _, buffer = cv2.imencode('.jpg', annotated_img_classes)
            processed_img_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', mask)
            mask_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', square_image)
            square_image_encoded = base64.b64encode(buffer).decode('utf-8')

            _, buffer = cv2.imencode('.jpg', classification_image)
            classification_image_encoded = base64.b64encode(buffer).decode('utf-8')

            images_encoded = []
            for cropped_image, classification, (start_x, start_y, end_x, end_y, center_x, center_y) in extracted_cells:
                _, buffer = cv2.imencode('.jpg', cropped_image)
                img_encoded = base64.b64encode(buffer).decode('utf-8')
                # Save the image and its metadata to the database
                save_image_data(base64.b64decode(img_encoded), classification, start_x, start_y, end_x, end_y, center_x, center_y, '0.40', datetime.now().strftime('%Y-%m-%d'))

                images_encoded.append(img_encoded)

            return jsonify({
                'original_image': original_img_encoded,
                'processed_image': processed_img_encoded,
                'mask': mask_encoded,
                'square_image': square_image_encoded,
                'classification_image': classification_image_encoded,
                'classifications': classifications,
                'probabilities': probabilities,
                'extracted_cells': images_encoded
            })

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)