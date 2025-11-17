from pathlib import Path
import numpy as np
from cellpose import models
import imaging_server_kit as sk


@sk.algorithm(
    name="CellPose",
    parameters={
        "image": sk.Image(
            name="Image",
            description="Input image (2D).",
            dimensionality=[2],
        ),
        "model_name": sk.Choice(
            name="Model",
            description="The model used for instance segmentation",
            items=["cyto", "nuclei", "cyto2"],
            default="cyto",
        ),
        "diameter": sk.Integer(
            name="Cell diameter (px)",
            description="The approximate size of the objects to detect",
            default=20,
            min=0,
            max=100,
            step=1,
        ),
        "flow_threshold": sk.Float(
            name="Flow threshold",
            description="The flow threshold",
            default=0.3,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        "cellprob_threshold": sk.Float(
            name="Probability threshold",
            description="The detection probability threshold",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
        "gpu": sk.Bool(name="GPU", default=False),
    },
    description="A generalist algorithm for cellular segmentation.",
    tags=[
        "Segmentation",
        "Deep learning",
        "Fluorescence microscopy",
        "Digital pathology",
        "Cell biology",
    ],
    project_url="https://github.com/MouseLand/cellpose",
    samples=[{"image": str(Path(__file__).parent / "samples" / "nuclei_2d.tif")}],
)
def cellpose_algo(
    image: np.ndarray,
    model_name: str,
    diameter: int,
    flow_threshold: float,
    cellprob_threshold: float,
    gpu: bool,
):
    model = models.CellposeModel(
        gpu=gpu,
        model_type=model_name,
    )

    if diameter == 0:
        diameter = None

    segmentation, flows, styles = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],  # Grayscale image only (for now)
    )
    return sk.Mask(segmentation, name="CellPose result")


if __name__ == "__main__":
    sk.serve(cellpose_algo)
