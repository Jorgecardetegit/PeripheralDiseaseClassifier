from transformers import AutoModel, AutoTokenizer
import torch

from transformers import ConvNextForImageClassification

# Load pre-trained model and tokenizer


model_id ="JorgeGIT/finetuned-Leukemia-cell"  # replace with your model name
model = ConvNextForImageClassification.from_pretrained(model_id)

# Set the model to inference mode
model.eval()

# Create a dummy input corresponding to the expected input size (e.g., 224x224 RGB image)
# The exact size and input dimension might need to be adjusted based on the model's requirements
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 pixels

# Export the model to ONNX
torch.onnx.export(model,               # the model to be exported
                  dummy_input,         # model input (dummy data)
                  "convnext_model.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=13,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to perform constant folding for optimization
                  input_names=['input'],      # the model's input names
                  output_names=['output'],    # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})