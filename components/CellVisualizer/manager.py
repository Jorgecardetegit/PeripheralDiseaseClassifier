from wasabi import msg

from components.CellVisualizer.visualizer import CellVisualizer
from components.CellVisualizer.interface import Visualizer

class VisualizerManager:
    def __init__(self):
        self.visualizers: dict[str, Visualizer] = {
            "visualizer": CellVisualizer(),
        }
        self.selected_visualizer: Visualizer= self.visualizers["visualizer"]
    
    def set_visualizer(self, visualizer: str) -> bool:
        if visualizer in self.visualizers:
            self.selected_visualizer = self.visualizers[visualizer]
            return True
        else:
            msg.warn(f"Visualizer {visualizer} not found")
            return False
    
    def get_visualizers(self) -> dict[str, Visualizer]:
        return self.selected_visualizer
    
