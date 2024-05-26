from components.grayscaleConverter.manager import grayscaleConverterManager
from components.grayscaleConverter.interface import Converter

from components.binaryMask.manager import binaryMaskManager
from components.binaryMask.interface import BinaryMasker

from components.objectsLocation.manager import ObjectsLocationManager
from components.objectsLocation.interface import objectsLocation

from components.sizeMask.manager import SizeMaskerManager
from components.sizeMask.interface import SizeMask

from components.cellExtractor.manager import cellExtractorManager
from components.cellExtractor.interface import Extractor

from components.CellClassifier.manager import cellClassifierManager
from components.CellClassifier.interface import Classifier

from components.CellVisualizer.manager import VisualizerManager
from components.CellVisualizer.interface import Visualizer

from wasabi import msg
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

class PeripheralManager: 
    """Manages all the app components."""

    def __init__(self) -> None:
        self.grayscaleConverter_manager = grayscaleConverterManager()
        self.mask_manager = binaryMaskManager()
        self.ObjectsLocation_manager = ObjectsLocationManager()
        self.SizeMasker_manager = SizeMaskerManager()
        self.cellExtractor_manager = cellExtractorManager()
        self.classifier_manager = cellClassifierManager()
        self.visualizer_manager = VisualizerManager()

        """Simple mask parameters"""
        self.threshold = 60

        """Localizer parameters"""
        self.offset = 10 

        """Visualizer parameters"""
        self.box_thickness=4, 
        self.fixed_size=500

        """CellExtractor parameters"""
        self.fixed_size=300

        """Selections"""
        self.set_converter = "simple"
        self.set_mask = "simple"
        self.set_localizer = "simple"
        self.set_sizemasker = "SizeMasker"
        self.set_extractor = "extractor"
        self.set_classifier = "ConvNext"
        self.set_visualizer = "visualizer"

    def convert_grayscale(self, 
                          image: np.ndarray, 
                          converter: str
                          ) -> dict[str, Converter]:

        converter = self.grayscaleConverter_manager.set_converter(converter)
        if converter == True: 
            selected_converter = self.grayscaleConverter_manager.get_converter()

        else:
            selected_converter = self.grayscaleConverter_manager.selected_converter
        
        return selected_converter.enhance_purple_to_grayscale(image)

        
    def binary_mask(self, 
                    grayscale_image: np.ndarray, 
                    masker: str
                    ) -> dict[str, BinaryMasker]:

        binary_masker = self.mask_manager.set_binaryMask(masker)
        if binary_masker == True:
            selected_masker = self.mask_manager.get_binaryMask()

        else: 
            selected_masker = self.mask_manager.selected_masker
            

        if masker != "adaptative":
            return selected_masker.binary_mask(grayscale_image, self.threshold)
        
        else:
            return selected_masker.binary_mask(grayscale_image)
        
    
    def cell_localizers(self, 
                        binary_image, 
                        localizer: str
                        ) -> dict[str, objectsLocation]:

        localizer = self.ObjectsLocation_manager.set_localizer(localizer)
        if localizer == True:
            selected_localizer = self.ObjectsLocation_manager.get_localizers()

        else: 
            selected_localizer = self.ObjectsLocation_manager.selected_localizer

        return selected_localizer.find_objects_locations(binary_image, self.offset)
    
    def size_mask(self, 
                  locations: dict[str, SizeMask], 
                  masker: str
                  ) -> dict[str, SizeMask]:

        size_masker = self.SizeMasker_manager.set_sizeMasker(masker)
        if size_masker == True:
            selected_sizeMasker = self.SizeMasker_manager.get_sizeMasker()

        else: 
            selected_sizeMasker = self.SizeMasker_manager.selected_sizemasker

        return selected_sizeMasker.size_mask(locations)
    
    def classification(self, 
                       image: np.ndarray, 
                       filtered_locations: dict[str, SizeMask], 
                       classifier: str
                       ) -> Tuple[List[int], List[List[Tuple[int, float]]]]:
        
        classifier = self.classifier_manager.set_classifier(classifier)
        if classifier == True:
            selected_classifier = self.classifier_manager.get_classifiers()
        
        else:
            selected_classifier = self.classifier_manager.selected_classifier

        return selected_classifier.classify_cells(image, filtered_locations)
    
    def main_visualizer(self, 
                   image: np.ndarray, 
                   binary_mask: np.ndarray,
                   bounding_boxes: dict[str, SizeMask], 
                   classifications: Tuple[List[int], List[List[Tuple[int, float]]]],  
                   visualizer: str,
                   type: str
                   ) -> dict[str, Visualizer]:
        
        visualizer = self.visualizer_manager.set_visualizer(visualizer)
        if visualizer == True:
            selected_visualizer = self.visualizer_manager.get_visualizers()
        
        else:
            selected_visualizer = self.visualizer_manager.selected_visualizer
        
        if type == "everything": 
            return selected_visualizer.visualize(image, bounding_boxes, classifications)
        
        elif type == "mask": 
            return selected_visualizer.visualize_mask(image, binary_mask)
        
        elif type == "boxes": 
            return selected_visualizer.visualize_bounding_boxes(image, bounding_boxes, classifications)
        
        elif type == "visualizations": 
            return selected_visualizer.visualize_classifications(image, bounding_boxes, classifications)
        
    def cellExtractor(self, 
                      image: np.ndarray, 
                      filtered_locations: dict[str, SizeMask],
                      classifications: Tuple[List[int], List[List[Tuple[int, float]]]], 
                      extractor: str
                      ) -> dict[str, Extractor]:

        cellExtractor = self.cellExtractor_manager.set_extractor(extractor)
        if cellExtractor == True:
            selected_extractor = self.cellExtractor_manager.get_extractors()

        else: 
            selected_extractor = self.cellExtractor_manager.selected_extractor

        return selected_extractor.extract_cells(image, filtered_locations, classifications)
    
    def main_processor(self, 
             image: np.ndarray):
        
        grayscale_image = self.convert_grayscale(image, self.set_converter)

        binary_mask = self.binary_mask(grayscale_image, self.set_mask)

        locations = self.cell_localizers(binary_mask, self.set_localizer)

        filtered_locations = self.size_mask(locations, self.set_sizemasker)

        classifications = self.classification(image, filtered_locations, self.set_classifier)

        return grayscale_image, binary_mask, filtered_locations, classifications 
    
    def cell_visualizer(self, 
                        image: np.ndarray, 
                        binary_mask: np.ndarray,
                        filtered_locations, 
                        classifications, 
                        set_visualizer): 
        
        eveything_image = self.main_visualizer(image, binary_mask, filtered_locations, classifications, "visualizer", "everything")

        binary_mask = self.main_visualizer(image, binary_mask, filtered_locations, classifications,"visualizer", "mask")

        boxes = self.main_visualizer(image, binary_mask, filtered_locations, classifications, "visualizer", "boxes")

        visualizations = self.main_visualizer(image, binary_mask, filtered_locations, classifications, "visualizer", "visualizations")

        return eveything_image, binary_mask, boxes, visualizations 
    
    