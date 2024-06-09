from wasabi import msg

from components.zoomer.zoomer_simple import Zoomer
from components.zoomer.interface import Zoom

class ZoomerManager:
    def __init__(self):
        self.zoomers: dict[str, Zoom] = {
            "Simple_zoomer": Zoom(),
        }
        self.selected_zoomer: Zoom = self.zoomers["Simple_zoomer"]
    
    def set_zoomer(self, zoomer: str) -> bool:
        if zoomer in self.zoomers: 
            self.selected_sizemasker = self.zoomers[zoomer]
            return True
        else:
            msg.warn(f"Zoomer {zoomer} not found")
            return False
    
    def get_zoomer(self) -> dict[str, Zoomer]:
        return self.selected_zoomer
    