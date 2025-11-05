from pathlib import Path
import numpy as np
from pystackreg import StackReg
import imaging_server_kit as sk

reg_type_dict = {
    "affine": StackReg.AFFINE,
    "rigid": StackReg.RIGID_BODY,
}


@sk.algorithm(
    name="pyStackReg",
    description="Register one or more images to a common reference image.",
    project_url="https://pystackreg.readthedocs.io/en/latest/",
    parameters={
        "image_stack": sk.Image(
            name="Image stack",
            description="2D image stack in (T)ZYX order.",
            dimensionality=[3],
        ),
        "reg_type": sk.Choice(
            name="Type",
            description="Registration type.",
            items=list(reg_type_dict.keys()),
            default="rigid",
        ),
        "reference": sk.Choice(
            name="Reference",
            description="Reference for the registration",
            items=["previous", "first", "mean"],
            default="previous",
        ),
        "axis": sk.Integer(
            name="Axis",
            description="Registration axis.",
            default=0,
            min=0,
            max=2,
        ),
    },
    samples=[{"image_stack": Path(__file__).parent / "samples" / "pc12-unreg.tif"}],
)
def stackreg_algo(
    image_stack: np.ndarray,
    reg_type: str,
    reference: str,
    axis: int,
):
    """Run the Pystackreg algorithm."""
    sr = StackReg(reg_type_dict.get(reg_type))
    tmats = sr.register_stack(
        image_stack,
        reference=reference,
        axis=axis,
    )
    registered_stack = sr.transform_stack(image_stack, tmats=tmats)
    return sk.Image(
        registered_stack,
        name="Registered stack",
        meta={
            "colormap": "viridis",
            "contrast_limits": [registered_stack.min(), registered_stack.max()],
        },
    )


if __name__ == "__main__":
    sk.serve(stackreg_algo)
