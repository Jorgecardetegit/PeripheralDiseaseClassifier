from wasabi import msg

from components.sizeMask.sizeMasker import SizeMasker
from components.sizeMask.interface import SizeMask

class SizeMaskerManager:
    def __init__(self):
        self.size_maskers: dict[str, SizeMask] = {
            "SizeMasker": SizeMasker(),
        }
        self.selected_sizemasker: SizeMasker = self.size_maskers["SizeMasker"]
    
    def set_sizeMasker(self, sizemasker: str) -> bool:
        if sizemasker in self.size_maskers: 
            self.selected_sizemasker = self.size_maskers[sizemasker]
            return True
        else:
            msg.warn(f"Size masker {sizemasker} not found")
            return False
    
    def get_sizeMasker(self) -> dict[str, SizeMasker]:
        return self.selected_sizemasker
    
