from wasabi import msg

from components.binaryMask.simpleMask import simpleMask
from components.binaryMask.adaptativeMask import adaptativeMask
from components.binaryMask.interface import BinaryMasker

class binaryMaskManager:
    def __init__(self):
        self.maskers: dict[str, BinaryMasker] = {
            "adaptative": adaptativeMask(),
            "simple": simpleMask()
        }
        self.selected_masker: BinaryMasker = self.maskers["simple"]
    
    def set_binaryMask(self, masker: str) -> bool:
        if masker in self.maskers:
            self.selected_masker = self.maskers[masker]
            return True
        else:
            msg.warn(f"Embedder {masker} not found")
            return False
    
    def get_binaryMask(self) -> dict[str, BinaryMasker]:
        return self.selected_masker
