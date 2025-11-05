import numpy as np
import rembg
import imaging_server_kit as sk
import skimage.data

sessions: dict[str, rembg.sessions.BaseSession] = {}


@sk.algorithm(
    name="Rembg",
    parameters={
        "image": sk.Image(
            name="Image",
            description="Input image (2D grayscale, RGB)",
            rgb=True,
        ),
        "model_name": sk.Choice(
            name="Model",
            description="The model used for background removal.",
            items=["silueta", "u2net"],
            default="silueta",
        ),
    },
    samples=[{"image": skimage.data.astronaut()}],
    description="A tool to remove images background.",
    tags=["Segmentation", "Deep learning"],
    project_url="https://github.com/danielgatis/rembg",
)
def rembg_algo(image, model_name):
    """Binary segmentation using rembg."""
    session = sessions.setdefault(model_name, rembg.new_session(model_name))
    image = image.astype(np.uint8)

    segmentation = rembg.remove(
        data=image,
        session=session,
        only_mask=True,
        post_process_mask=True,
    )

    mask = segmentation == 255

    return sk.Mask(mask, name=f"{model_name} result")


if __name__ == "__main__":
    sk.serve(rembg_algo)
