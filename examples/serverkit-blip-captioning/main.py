import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import imaging_server_kit as sk
import skimage.data


@sk.algorithm(
    parameters={
        "image": sk.Image(description="Input image (2D, RGB)."),
        "conditional_text": sk.String(
            name="Conditional text",
            description="Conditional text (beginning of the caption).",
            default="an image of",
        ),
    },
    samples=[{"image": skimage.data.astronaut()}],
)
def blip_algo(image: np.ndarray, conditional_text: str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    inputs = processor(image, conditional_text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return sk.String(caption, name="Caption")


if __name__ == "__main__":
    sk.serve(blip_algo)
