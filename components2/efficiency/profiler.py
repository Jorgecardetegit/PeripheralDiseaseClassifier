import cProfile
import pstats
import io
import csv
from io import BytesIO

from main import process_image, visualize_results, load_image_from_file, extract_cells

def main():
    with open(r"C:\Users\JCardeteLl\Documents\TFG\Bases de datos\FOTOS JORGE\L. VELLOSOS\0001-2.JPG", 'rb') as file:
        file_content = file.read()  # Read file content into memory
        file_stream = BytesIO(file_content)  # Create a BytesIO object

    image = load_image_from_file(file_stream)

    binary_img, filtered_objects, classifications, probabilities = process_image(image)
    annotated_img, mask, bounding_boxes, classified_img = visualize_results(image, binary_img, filtered_objects, classifications)
    extracted_cells = extract_cells(image, filtered_objects, classifications)

def write_profiling_to_csv(stats, filename='profiling_output.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Function', 'Total Calls', 'Total Time', 'Cumulative Time', 'Per Call Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for func, (cc, nc, tt, ct, callers) in sorted(stats.stats.items()):
            writer.writerow({
                'Function': func,
                'Total Calls': nc,
                'Total Time': tt,
                'Cumulative Time': ct,
                'Per Call Time': ct / nc if nc else 0
            })

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    write_profiling_to_csv(stats)
