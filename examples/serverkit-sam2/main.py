from pathlib import Path
import numpy as np
import pandas as pd
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import imaging_server_kit as sk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2-hiera-tiny", device=DEVICE
)
generator = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-tiny", device=DEVICE
)


@sk.algorithm(
    parameters={
        "image": sk.Image(description="Input image (2D, RGB).", dimensionality=[2, 3]),
        "boxes": sk.Boxes(
            name="Boxes",
            description="Boxes prompt.",
            required=False,
        ),
        "points": sk.Points(
            name="Points",
            description="Points prompt.",
            required=False,
        ),
        "auto_mode": sk.Bool(
            name="Auto mode",
            description="Run SAM in auto (grid) mode",
            default=False,
        ),
    },
    samples=[
        {
            "image": Path(__file__).parent / "samples" / "groceries.jpg",
            "points": pd.read_csv(Path(__file__).parent / "samples" / "points.csv")[
                ["axis-0", "axis-1"]
            ].values,
        },
    ],
)
def sam2_algo(image, boxes, points, auto_mode):
    if len(image.shape) == 2:
        image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        image = gray2rgb(image)

    rx, ry, _ = image.shape
    mask = np.zeros((rx, ry), dtype=np.uint16)

    if auto_mode:
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            masks = generator.generate(image)

        for k, ann in enumerate(masks):
            m = ann.get("segmentation")
            mask[m] = k + 1
    else:
        # Boxes should be in (N, 4) format (top-left, bottom-right corner)
        # Invert X-Y and keep only two vertices
        # TODO: This causes errors if the boxes arent drawn in the top to bottom direction
        if boxes is not None:
            if len(boxes) > 0:
                boxes = boxes[:, ::2, ::-1].reshape((len(boxes), -1)).copy()

        # Handle points prompts
        point_labels = None
        if points is not None:
            if len(points) > 0:
                point_labels = np.ones(len(points))
                points = points[:, ::-1].copy()  # Invert X-Y

        # Run the model
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=boxes,
                multimask_output=False,
            )

        for k, m in enumerate(masks):
            mask[np.squeeze(m) == 1] = k + 1

    return sk.Mask(mask, name="SAM-2 result")


if __name__ == "__main__":
    sk.serve(sam2_algo)
