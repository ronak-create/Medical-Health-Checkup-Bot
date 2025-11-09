from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

# Load an image
image_path = "test3.png"
image = Image.open(image_path)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Get predicted label
logits = outputs.logits
predicted_class = torch.argmax(logits).item()
labels = model.config.id2label
predicted_label = labels[predicted_class]


print(f"Predicted Expression: {predicted_label}")