import os
from ultralytics import YOLO

class ModelHandler:
    """
    Handles loading YOLO models from either PyTorch (.pt) or ONNX (.onnx) formats.
    If the ONNX model is not available, it exports it from the given PyTorch model.
    """

    def __init__(self, pt_model_path: str, onnx_model_path: str):
        """
        Initialize the model handler with paths to the PyTorch and ONNX models.
        
        Args:
            pt_model_path (str): Path to the PyTorch (.pt) YOLO model.
            onnx_model_path (str): Path to the ONNX (.onnx) YOLO model.
        """
        self.pt_model_path = pt_model_path
        self.onnx_model_path = onnx_model_path

    def load_model(self):
        """
        Load the YOLO model in ONNX format.
        If the ONNX file does not exist, export it from the PyTorch model.
        
        Returns:
            YOLO: Loaded YOLO model ready for inference.
        """
        # Check if ONNX model exists, otherwise export from .pt
        if not os.path.exists(self.onnx_model_path):
            # Load PyTorch YOLO model
            model = YOLO(self.pt_model_path)
            # Export to ONNX with dynamic shape and opset version 12
            model.export(format="onnx", opset=12, dynamic=True)
            print(f"ONNX model saved to {self.onnx_model_path}")

        # Load the ONNX model
        model = YOLO(self.onnx_model_path)

        # If class names are missing, assign default mapping
        if not model.names or len(model.names) == 0:
            model.names = {0: "worker", 1: "mobile"}

        print("Class mapping:", model.names)
        return model