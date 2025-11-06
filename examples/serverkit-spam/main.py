import imaging_server_kit as sk
import spam.DIC
import spam.deformation
import numpy as np
from pathlib import Path


@sk.algorithm(
    parameters={
        "fixed_image": sk.Image(
            name="Fixed image",
            description="Fixed image (2D, 3D).",
            dimensionality=[2, 3],
        ),
        "moving_image": sk.Image(
            name="Moving image",
            description="Moving image (2D, 3D).",
            dimensionality=[2, 3],
        ),
    },
    samples=[
        {
            "fixed_image": Path(__file__).parent / "samples" / "image1.tif",
            "moving_image": Path(__file__).parent / "samples" / "image2.tif",
        }
    ],
)
def spam_reg_algo(fixed_image, moving_image):
    reg = spam.DIC.register(moving_image, fixed_image)

    Phi = reg.get("Phi")
    error = reg.get("deltaPhiNorm")

    tform = spam.deformation.decomposePhi(Phi)

    dy = tform["t"][1]
    dx = tform["t"][2]

    disp_y = np.ones_like(fixed_image) * dy
    disp_x = np.ones_like(fixed_image) * dx
    # err_img = np.ones_like(fixed_image) * error

    ry, rx = fixed_image.shape
    src_pt = np.array([[ry / 2, rx / 2]])
    disp = np.array([[dy, dx]])
    vectors = np.stack((src_pt, disp), axis=1)

    vectors_meta = {
        "vector_style": "arrow",
        "features": {"error": error},
        "edge_color": "blue",
    }

    registered_image = spam.DIC.applyPhiPython(moving_image, Phi)

    return (
        sk.Image(
            registered_image,
            name="Registered",
            meta={"contrast_limits": [float(registered_image.min()), float(registered_image.max())]},
        ),
        sk.Image(disp_y, name="Dy", meta={"colormap": "viridis"}),
        sk.Image(disp_x, name="Dx", meta={"colormap": "viridis"}),
        # sk.Image(err_img, name="Error", meta={"colormap": "viridis"}),
        sk.Vectors(vectors, name="Displacements", meta=vectors_meta),
    )


if __name__ == "__main__":
    sk.serve(spam_reg_algo)