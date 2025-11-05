import imaging_server_kit as sk
from stardist.models import StarDist2D


def normalize(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr


@sk.algorithm(
    name="stardist",
    title="Streaming tiles - StarDist 2D example",
    parameters={
        "image": sk.sk.Image(dimensionality=[2, 3]),
    },
    samples=["/home/wittwer/data/test_images/C3-bigimg-gray-inv-crop.tif"],
)
def stream_stardist_2d(image):
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    segmentation, polys = model.predict_instances(
        normalize(image),
        prob_thresh=0.5,
        nms_thresh=0.4,
        scale=1.0,
    )

    if segmentation.max() > 0:
        return [
            (
                segmentation,
                {"name": "Stardist result"},
                "instance_mask",
            )
        ]
