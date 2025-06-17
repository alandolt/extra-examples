from typing import List
from pathlib import Path
import numpy as np
import uvicorn
from instanseg import InstanSeg

from imaging_server_kit import algorithm_server, ImageUI, DropDownUI, FloatUI


@algorithm_server(
    algorithm_name="instanseg",
    parameters={
        "image": ImageUI(description="Input image (2D, RGB)."),
        "model_name": DropDownUI(
            default="fluorescence_nuclei_and_cells",
            items=["fluorescence_nuclei_and_cells", "brightfield_nuclei"],
            title="Model",
            description="Segmentation model",
        ),
        "pixel_size": FloatUI(
            default=0.5,
            title="Pixel size (um/px)",
            description="Pixel size (um/px)",
            min=0.1,
            max=10.0,
            step=0.05,
        ),
    },
    sample_images=[
        Path(__file__).parent / "sample_images" / "Fluorescence_example.tif"
    ],
    metadata_file="metadata.yaml",
)
def instanseg_server(
    image: np.ndarray,
    model_name: str,
    pixel_size: float,
) -> List[tuple]:
    """Runs the algorithm."""
    instanseg_brightfield = InstanSeg(model_name, image_reader="tiffslide", verbosity=1)

    segmentation, image_tensor = instanseg_brightfield.eval_small_image(
        image, pixel_size
    )
    segmentation = segmentation[0].cpu().numpy()

    segmentation = segmentation[
        ::-1
    ]  # Apparently the channel axis needs to be inverted?

    segmentation_params = {"name": f"InstanSeg ({model_name})"}

    if model_name == "fluorescence_nuclei_and_cells":
        # The fluorescence model returns a 3D segmentation array for the 2D+c image
        return [(segmentation, segmentation_params, "mask3d")]
    else:
        # The brightfield nuclei model can be returned as Shapely features
        segmentation = np.squeeze(segmentation)
        return [(segmentation, segmentation_params, "instance_mask")]


if __name__ == "__main__":
    uvicorn.run(instanseg_server.app, host="0.0.0.0", port=8000)
