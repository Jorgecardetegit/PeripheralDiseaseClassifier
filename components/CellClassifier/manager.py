from wasabi import msg

from components.CellClassifier.ConvNext import ConvNext
from components.CellClassifier.efficientnet import efficientnet

from components.CellClassifier.interface import Classifier


class cellClassifierManager:
    def __init__(self):
        self.classifiers: dict[str, Classifier] = {
            "ConvNext": ConvNext(),
            "efficientnet": efficientnet()
        }
        self.selected_classifier: Classifier = self.classifiers["ConvNext"]
    
    def set_classifier(self, classifier: str) -> bool:
        if classifier in self.classifiers:
            self.selected_classifier= self.classifiers[classifier]
            return True
        else:
            msg.warn(f"Classifier {classifier} not found")
            return False
    
    def get_classifiers(self) -> dict[str, Classifier]:
        return self.selected_classifier
    

