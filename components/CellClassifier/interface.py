from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Classifier(ABC):

    def __init__(self):
        super().__init__()
        self.class_names = {
            1: "Chronic lymphocytic leukemia (CLL)",
            2: "Tricholeukemia",
            3: "Acute lymphoblastic leukemia (ALL)",
            4: "Follicular lymphoma",
            5: "Marginal lymphoma",
            6: "Mononucleosis",
            7: "Normal"
            }        
        
    @abstractmethod
    def classify_cells(self) -> List[Tuple[np.ndarray, str, Tuple[int, int, int, int, int, int]]]:
        pass