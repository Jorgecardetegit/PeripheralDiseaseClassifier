from transformers import AutoModelForImageClassification, ConvNextImageProcessor
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the model and the feature extractor
model_id = "JorgeGIT/finetuned-Leukemia-cell"
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = ConvNextImageProcessor.from_pretrained(model_id)

# Load and process an image
image_path = r"C:\Users\JCardeteLl\Documents\TFG\Bases de datos\FOTOS JORGE\L.A. LINFOBLµSTICA\image_2012y06m04d_14h40m17s.jpg"  # Change this to your image path
image = Image.open(image_path).convert('RGB')  # Convert to RGB if necessary
inputs = processor(images=image, return_tensors="pt")
img_tensor = inputs['pixel_values']

for name, module in model.named_modules():
    print(name)

def grad_cam(model, img_tensor, target_layer, target_class_idx):
    model.eval()  # Set the model to evaluation mode
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        return None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
        return None

    # Attach hooks
    target_layer_module = dict([*model.named_modules()])[target_layer]
    forward_handle = target_layer_module.register_forward_hook(forward_hook)
    backward_handle = target_layer_module.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(img_tensor)
    pred_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_pred_prob, top_pred_label = pred_probabilities.topk(1)
    target_class_idx = target_class_idx if target_class_idx is not None else top_pred_label.item()

    # Backward pass
    model.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs.logits, torch.tensor([target_class_idx], device=img_tensor.device))
    loss.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    activations *= pooled_gradients
    heatmap = torch.mean(activations, dim=1).squeeze().detach()
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Using the correct layer
heatmap = grad_cam(model, img_tensor, target_layer='convnext.encoder.stages.3.layers.2.dwconv', target_class_idx=None)

# Visualization
plt.matshow(heatmap, cmap='jet')
plt.colorbar()
plt.show()