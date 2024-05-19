import cv2
from components.objectsLocation.interface import objectsLocation

class objects_localizer(objectsLocation): 
    def __init__(self):
        super().__init__()
        pass

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

