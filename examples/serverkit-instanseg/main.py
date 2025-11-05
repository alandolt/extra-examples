from pathlib import Path
import numpy as np
from instanseg import InstanSeg
import imaging_server_kit as sk


@sk.algorithm(
    parameters={
        "image": sk.Image(description="Input image (2D, RGB)."),
        "model_name": sk.Choice(
            name="Model",
            description="Segmentation model",
            items=["fluorescence_nuclei_and_cells", "brightfield_nuclei"],
            default="fluorescence_nuclei_and_cells",
        ),
        "pixel_size": sk.Float(
            name="Pixel size (um/px)",
            description="Pixel size (um/px)",
            default=0.5,
            min=0.1,
            max=10.0,
            step=0.05,
        ),
    },
    samples=[{"image": Path(__file__).parent / "samples" / "Fluorescence_example.tif"}],
)
def instanseg_algo(image, model_name, pixel_size):
    model_bf = InstanSeg(model_name, image_reader="tiffslide", verbosity=1)

    mask, image_tensor = model_bf.eval_small_image(image, pixel_size)
    mask = mask[0].cpu().numpy()
    mask = mask[::-1]  # Channel axis inverted?

    if model_name != "fluorescence_nuclei_and_cells":
        mask = np.squeeze(mask)

    return sk.Mask(mask, name=f"InstanSeg ({model_name})")


if __name__ == "__main__":
    sk.serve(instanseg_algo)
