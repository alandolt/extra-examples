from typing import List
from pathlib import Path
import numpy as np
import uvicorn
from spotiflow.model import Spotiflow

from imaging_server_kit import (
    algorithm_server,
    ImageUI,
    FloatUI,
    IntUI,
    DropDownUI,
)


@algorithm_server(
    algorithm_name="spotiflow",
    parameters={
        "image": ImageUI(
            title="Image (YX or CYX)",
            description="Input image. If 2D, interpreted as single-channel (YX order). If 3D, interpreted as multichannel in CYX order.",
            dimensionality=[2, 3],
        ),
        "model_name": DropDownUI(
            title="Model",
            description="Pretrained model to use.",
            items=["general", "hybiss"],
            default="general",
        ),
        "prob_thresh": FloatUI(
            default=0.5,
            min=0,
            max=1,
            step=0.01,
            title="Probability threshold",
            description="Probability threshold for detection.",
        ),
        "min_distance": IntUI(
            default=1,
            min=0,
            max=100,
            title="Min distance",
            description="Minimum distance between spots for NMS.",
        ),
        "device": DropDownUI(
            title="Device",
            description="Device to use for inference",
            items=["cpu", "cuda", "mps"],
            default="cpu",
        ),
    },
    sample_images=[
        Path(__file__).parent / "sample_images" / "hybiss_2d.tif",
    ],
    metadata_file="metadata.yaml",
)
def spotiflow_server(
    image: np.ndarray,
    model_name: str,
    prob_thresh: float,
    min_distance: int,
    device: str,
) -> List[tuple]:
    """Runs the algorithm."""
    model = Spotiflow.from_pretrained(model_name, map_location=device)

    points_meta = {
        "name": f"Detected spots ({model_name} model)",
        "opacity": 0.7,
        "face_color": "intensities",
        "border_color": "transparent",
        "size": 5,
    }

    # Single-channel 2D case
    if image.ndim == 2:
        points, details = model.predict(
            image,
            prob_thresh=prob_thresh,
            min_distance=min_distance,
        )

        probabilities = details.prob
        intensities = np.squeeze(details.intens)

        points_meta["features"] = {
            "probabilities": probabilities,
            "intensities": intensities,
        }

        return [(points, points_meta, "points")]

    # Multi-channel 2D case (CYX order): we predict all channels independently and aggregate the results
    elif image.ndim == 3:
        all_points = []
        all_probabilities = []
        all_intensities = []
        for channel_idx, channel in enumerate(image):
            points, details = model.predict(
                channel,
                prob_thresh=prob_thresh,
                min_distance=min_distance,
            )

            probabilities = details.prob
            all_probabilities.append(probabilities)

            intensities = np.squeeze(details.intens)
            all_intensities.append(intensities)

            # Add channel dimension to points
            points3d = np.hstack((np.ones((len(points), 1)) * channel_idx, points))
            all_points.append(points3d)

        all_points = np.vstack([*all_points])
        all_intensities = np.hstack([*all_intensities])
        all_probabilities = np.hstack([*all_probabilities])

        points_meta["features"] = {
            "probabilities": all_probabilities,
            "intensities": all_intensities,
        }

        return [(all_points, points_meta, "points3d")]


if __name__ == "__main__":
    uvicorn.run(spotiflow_server.app, host="0.0.0.0", port=8000)
