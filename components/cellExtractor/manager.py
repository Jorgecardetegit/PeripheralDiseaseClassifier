from wasabi import msg

from components.cellExtractor.extractor import cellExtractor
from components.cellExtractor.interface import Extractor

class cellExtractorManager:
    def __init__(self):
        self.extractors: dict[str, Extractor] = {
            "extractor": cellExtractor(),
        }
        self.selected_extractor: Extractor = self.extractors["extractor"]
    
    def set_extractor(self, extractor: str) -> bool:
        if extractor in self.extractors:
            self.selected_extractor = self.extractors[extractor]
            return True
        else:
            msg.warn(f"Extractor {extractor} not found")
            return False
    
    def get_extractors(self) -> dict[str, Extractor]:
        return self.selected_extractor
    
