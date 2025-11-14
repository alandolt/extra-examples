"""
Inspired from: https://splinebox.readthedocs.io/en/latest/auto_examples/plot_splinebox_vs_scipy_coin.html#sphx-glr-auto-examples-plot-splinebox-vs-scipy-coin-py
"""

import imaging_server_kit as sk
import numpy as np
import skimage
import splinebox
from skimage.draw import ellipse


def get_sample():
    img = np.zeros((800, 800))
    rr, cc = ellipse(250, 200, 120, 200, img.shape)
    img[rr, cc] = 1
    rr, cc = ellipse(550, 600, 150, 110, img.shape)
    img[rr, cc] = 1
    return img == 1


@sk.algorithm(
    name="Contour fitting",
    description="Mask contour approximation using a spline (splinebox).",
    project_url="https://splinebox.readthedocs.io/en/latest/index.html",
    parameters={
        "mask": sk.Mask(),
        "M": sk.Integer(name="M", min=4, max=100, default=9, auto_call=True),
        "basis": sk.Choice(
            name="Basis", items=["B1", "B2", "B3"], default="B3", auto_call=True
        ),
    },
    samples=[{"mask": get_sample()}]
)
def splinebox_algo(mask: sk.Mask, M, basis):
    contours = skimage.measure.find_contours(mask)

    basis_options = {
        "B1": splinebox.B1(),
        "B2": splinebox.B2(),
        "B3": splinebox.B3(),
    }

    splinebox_vals = []
    control_points = np.array((0, 2))
    for c in contours:
        spline = splinebox.Spline(M=M, basis_function=basis_options[basis], closed=True)
        try:
            spline.fit(c[:-1])
        except:
            yield sk.Notification(f"Could not fit spine on contour of size {len(c)}")
        splinebox_vals.append(spline(np.linspace(0, M, len(c))))
        control_points = np.vstack([control_points, spline.control_points])

        yield [
            sk.Paths(
                splinebox_vals,
                meta={
                    "face_color": "transparent",
                    "edge_color": "blue",
                    "edge_width": 1,
                },
            ),
            sk.Points(
                control_points[1:],
                name="Control points",
                meta={"size": 2, "face_color": "blue", "border_width": 0},
            ),
        ]


if __name__ == "__main__":
    sk.serve(splinebox_algo)
