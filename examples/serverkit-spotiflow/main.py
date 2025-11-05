from pathlib import Path
import numpy as np
from spotiflow.model import Spotiflow
import imaging_server_kit as sk


@sk.algorithm(
    parameters={
        "image": sk.Image(
            name="Image (YX or CYX)",
            description="Input image. If 2D, interpreted as single-channel (YX order). If 3D, interpreted as multichannel in CYX order.",
            dimensionality=[2, 3],
        ),
        "model_name": sk.Choice(
            name="Model",
            description="Pretrained model to use.",
            items=["general", "hybiss"],
            default="general",
        ),
        "prob_thresh": sk.Float(
            name="Probability threshold",
            description="Probability threshold for detection.",
            default=0.5,
            min=0,
            max=1,
            step=0.01,
        ),
        "min_distance": sk.Integer(
            name="Min distance",
            description="Minimum distance between spots for NMS.",
            default=1,
            min=0,
            max=100,
        ),
        "device": sk.Choice(
            name="Device",
            description="Device to use for inference",
            items=["cpu", "cuda", "mps"],
            default="cpu",
        ),
    },
    samples=[{"image": Path(__file__).parent / "samples" / "hybiss_2d.tif"}],
)
def spotiflow_algo(
    image: np.ndarray,
    model_name: str,
    prob_thresh: float,
    min_distance: int,
    device: str,
):
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

        return sk.Points(points, meta=points_meta)

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

        return sk.Points(all_points, meta=points_meta)


if __name__ == "__main__":
    sk.serve(spotiflow_algo)
