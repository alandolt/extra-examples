import numpy as np
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import imaging_server_kit as sk
import skimage.data

@sk.algorithm(
    name="ResNet50",
    parameters={"image": sk.Image(description="Input image (2D, RGB)")},
    samples=[{"image": skimage.data.astronaut()}],
)
def resnet_algo(image: np.ndarray):
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_idx]

    return sk.Choice(predicted_label)

if __name__ == "__main__":
    sk.serve(resnet_algo)
