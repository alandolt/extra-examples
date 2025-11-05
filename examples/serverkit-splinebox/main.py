"""
Inspired from: https://splinebox.readthedocs.io/en/latest/auto_examples/plot_splinebox_vs_scipy_coin.html#sphx-glr-auto-examples-plot-splinebox-vs-scipy-coin-py
"""

import imaging_server_kit as sk
import numpy as np
import skimage
import splinebox


@sk.algorithm(
    name="Contour fitting",
    description="Mask contour approximation using a spline (splinebox).",
    parameters={"M": sk.Integer(name="M", min=3, max=100, default=9, auto_call=True)},
    samples=[{"img": skimage.data.coins()[90:150, 240:300]}],
)
def splinebox_algo(img: sk.Image, M):
    thresh = skimage.filters.threshold_otsu(img)

    mask = img > thresh
    mask = skimage.morphology.remove_small_holes(mask)

    contours = skimage.measure.find_contours(mask)

    spline = splinebox.Spline(M=M, basis_function=splinebox.B3(), closed=True)
    spline.fit(contours[0][:-1])

    splinebox_vals = spline(np.linspace(0, M, 100))

    return (
        sk.Float(thresh),
        sk.Mask(mask),
        sk.Paths(
            contours,
            meta={"face_color": "transparent", "edge_color": "red", "edge_width": 1},
        ),
        sk.Paths(
            [splinebox_vals],
            meta={"face_color": "transparent", "edge_color": "blue", "edge_width": 1},
        ),
        sk.Points(
            spline.control_points,
            name="Control points",
            meta={"size": 2, "face_color": "blue", "border_width": 0},
        ),
    )


if __name__ == "__main__":
    sk.serve(splinebox_algo)