from pathlib import Path
import os
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import imaging_server_kit as sk

custom_model_path = "/models"  # Use relative path for portability
if os.path.exists(custom_model_path):
    custom_models = [f.name for f in os.scandir(custom_model_path) if f.is_dir()]
else:
    custom_models = []
base_models = ["2D_versatile_fluo", "2D_versatile_he"]  # Built-in StarDist models
models_list = base_models + custom_models
print(f"Available models: {models_list}")

# Global cache variables for performance
last_model = None
cached_model = None
last_custom_path = None
cached_custom_model = None


@sk.algorithm(
    name="StarDist (2D)",
    parameters={
        "image": sk.Image(
            description="Input image (2D).",
            dimensionality=[2, 3],
        ),
        "model_name": sk.Choice(
            name="Model",
            description="The model used for nuclei segmentation",
            items=models_list,  # Use dynamic list
            default="2D_versatile_fluo",
        ),
        "prob_thresh": sk.Float(
            name="Probability threshold",
            description="Predicted object probability threshold",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
        "nms_thresh": sk.Float(
            name="Overlap threshold",
            description="Overlapping objects are considered the same when their area/surface overlap exceeds this threshold",
            default=0.4,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
        "scale": sk.Float(
            name="Scale",
            description="Scale the input image internally by this factor and rescale the output accordingly (<1 to downsample, >1 to upsample)",
            default=1.0,
            min=0.0,
            max=1.0,
            step=0.1,
        ),
    },
    project_url="https://github.com/stardist/stardist",
    description="Object Detection with Star-convex Shapes.",
    tags=[
        "Segmentation",
        "Deep learning",
        "Fluorescence microscopy",
        "H&E",
        "Digital pathology",
        "Cell biology",
        "EPFL",
    ],
    samples=[
        {
            "image": str(Path(__file__).parent / "samples" / "nuclei_2d.tif"),
            "model_name": "2D_versatile_fluo",
        },
        {
            "image": str(Path(__file__).parent / "samples" / "BC9_7_img.png"),
            "model_name": "2D_versatile_he",
        },
    ],
)
def stardist_algo(
    image: np.ndarray,
    model_name: str,
    prob_thresh: float,
    nms_thresh: float,
    scale: float,
):
    """Instance cell nuclei segmentation using StarDist."""
    global last_model, cached_model, last_custom_path, cached_custom_model
    
    if model_name in custom_models:
        model_path = os.path.join(custom_model_path, model_name)
        if model_name != last_custom_path:
            print(f"Loading custom model from {model_path}")
            cached_custom_model = StarDist2D(None, name=model_name, basedir=custom_model_path)
            last_custom_path = model_name
        else:
            print(f"Using cached custom model from {model_path}")
        model = cached_custom_model
    else:
        if model_name != last_model:
            print(f"Loading built-in model: {model_name}")
            cached_model = StarDist2D.from_pretrained(model_name)
            last_model = model_name
        else:
            print(f"Using cached built-in model: {model_name}")
        model = cached_model

    if (image.shape[0] + image.shape[1]) / 2 < 1024:
        segmentation, polys = model.predict_instances(
            normalize(image),
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            scale=scale,
        )
    else:
        segmentation, polys = model.predict_instances_big(
            normalize(image),
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            scale=scale,
            block_size=512,
            min_overlap=64,
            axes="YX",
            return_labels=True,
        )

    return sk.Mask(segmentation, name=f"{model_name} result")


if __name__ == "__main__":
    sk.serve(stardist_algo)