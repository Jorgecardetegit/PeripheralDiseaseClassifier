from components.sizeMask.interface import SizeMask

class SizeMasker(SizeMask):
    def __init__(self):
        super().__init__()
        self.name = "sizeMasker"

    def size_mask(self, bounding_boxes) -> list:
        filtered_bounding_boxes = []
        
        for x, y, w, h in bounding_boxes:
            if (w*h >= bounding_boxes[-1][2]*bounding_boxes[-1][3] / 2):
                filtered_bounding_boxes.append((x, y, w, h))

        return filtered_bounding_boxes





