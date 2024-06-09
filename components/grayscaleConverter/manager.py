from wasabi import msg

from components.grayscaleConverter.simple_converter import SimpleConverter
from components.grayscaleConverter.custom_converter import CustomConverter
from components.grayscaleConverter.interface import Converter

class grayscaleConverterManager:
    def __init__(self):
        self.converters: dict[str, Converter] = {
            "simple": SimpleConverter(),
            "custom": CustomConverter()
        }
        self.selected_converter: Converter = self.converters["simple"]
    
    def set_converter(self, converter: str) -> bool:
        if converter in self.converters:
            self.selected_converter = self.converters[converter]
            return True
        else:
            msg.warn(f"Grayscale converte {converter} not found")
            return False
    
    def get_converter(self) -> dict[str, Converter]:
        return self.selected_converter
    
