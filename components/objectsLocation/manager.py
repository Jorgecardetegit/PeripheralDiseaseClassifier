from wasabi import msg

from components.objectsLocation.localizer import objects_localizer
from components.objectsLocation.interface import objectsLocation

class ObjectsLocationManager:
    def __init__(self):
        self.localizers: dict[str, objectsLocation] = {
            "simple": objects_localizer(),
        }
        self.selected_localizer: objectsLocation = self.localizers["simple"]
    
    def set_localizer(self, localizer: str) -> bool:
        if localizer in self.localizers:
            self.selected_localizer = self.localizers[localizer]
            return True
        else:
            msg.warn(f"Localizer {localizer} not found")
            return False
    
    def get_localizers(self) -> dict[str, objectsLocation]:
        return self.selected_localizer




