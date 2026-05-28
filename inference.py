import numpy as np
import torch


def prepare_image_for_model(image):
    image = image.resize((28, 28)).convert("L")
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image, image_tensor


def predict(model, class_names, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction_index = torch.argmax(probabilities).item()
    return class_names[prediction_index], probabilities[prediction_index].item()
