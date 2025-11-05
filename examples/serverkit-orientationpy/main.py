from pathlib import Path
import numpy as np
import orientationpy
import matplotlib
from scipy.ndimage import map_coordinates
import imaging_server_kit as sk


@sk.algorithm(
    name="Orientation analysis",
    parameters={
        "image": sk.Image(),
        "mode": sk.Choice(
            name="Mode",
            description="The orientation computation mode.",
            items=["fiber", "membrane"],
            default="fiber",
        ),
        "scale": sk.Float(
            name="Structural scale",
            description="The scale at which orientation is computed.",
            default=1.0,
            min=0.1,
            max=10.0,
            step=1.0,
        ),
        "vector_spacing": sk.Integer(
            name="Vector spacing",
            description="The spacing at which the orientation vectors are rendered.",
            default=3,
            min=1,
            max=10,
            step=1,
        ),
    },
    samples=[
        {
            "image": Path(__file__).parent
            / "sample_images"
            / "image1_from_OrientationJ.tif",
            "vector_spacing": 5,
        },
    ],
    description="Measure greyscale orientations from 2D and 3D images.",
    project_url="https://gitlab.com/epfl-center-for-imaging/orientationpy",
    tags=["Filtering", "EPFL"],
)
def orientationpy_algo(
    image: np.ndarray,
    mode: str,
    scale: float,
    vector_spacing: int,
):
    if image.size == 0:
        return sk.Notification("Image has size zero", meta={"level": "warning"})

    if image.ndim == 2:
        mode = "fiber"  # no membranes in 2D

    gradients = orientationpy.computeGradient(image, mode="splines")
    structureTensor = orientationpy.computeStructureTensor(gradients, sigma=scale)
    orientation_returns = orientationpy.computeOrientation(
        structureTensor,
        mode=mode,
    )
    theta = orientation_returns.get("theta") + 90
    phi = orientation_returns.get("phi")

    boxVectorCoords = orientationpy.anglesToVectors(orientation_returns)

    node_spacings = np.array([vector_spacing] * image.ndim).astype(int)
    slices = [slice(n // 2, None, n) for n in node_spacings]
    grid = np.mgrid[[slice(0, x) for x in image.shape]]
    node_origins = np.stack([g[tuple(slices)] for g in grid])
    slices.insert(0, slice(len(boxVectorCoords)))
    displacements = boxVectorCoords[tuple(slices)].copy()
    displacements *= np.mean(node_spacings)
    displacements = np.reshape(displacements, (image.ndim, -1)).T
    origins = np.reshape(node_origins, (image.ndim, -1)).T
    origins = origins - displacements / 2
    vectors = np.stack((origins, displacements))
    vectors = np.rollaxis(vectors, 1)

    if image.ndim == 3:
        imDisplayHSV = np.stack(
            (phi / 360, np.sin(np.deg2rad(theta)), image / image.max()), axis=-1
        )
    else:
        imDisplayHSV = np.stack(
            (theta / 180, np.ones_like(image), image / image.max()), axis=-1
        )
    imdisplay_rgb = matplotlib.colors.hsv_to_rgb(imDisplayHSV)
    
    vec_locs = vectors[:, 0][:, ::-1]
    
    def sample_image_at_coords(image, coords, order=1, mode='constant', cval=np.nan):
        sample_points = coords.T[::-1]
        return map_coordinates(image, sample_points, order=order, mode=mode, cval=cval)

    reds = (sample_image_at_coords(imdisplay_rgb[..., 0], vec_locs, order=1)*255).astype(np.uint8)
    greens = (sample_image_at_coords(imdisplay_rgb[..., 1], vec_locs, order=1)*255).astype(np.uint8)
    blues = (sample_image_at_coords(imdisplay_rgb[..., 2], vec_locs, order=1)*255).astype(np.uint8)
    alphas = np.ones(len(reds)) * 255

    edge_colors = np.stack((reds, greens, blues, alphas)).T
    
    meta = {
        "edge_width": np.max(node_spacings) / 3.0,
        "opacity": 1.0,
        "ndim": image.ndim,
        "vector_style": "line",
        "edge_color": edge_colors,
    }
    
    return sk.Vectors(vectors, name="Orientation vectors", meta=meta)


if __name__ == "__main__":
    sk.serve(orientationpy_algo)