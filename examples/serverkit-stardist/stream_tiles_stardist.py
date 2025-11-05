import imaging_server_kit as sk
from stardist.models import StarDist2D


def normalize(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr


@sk.algorithm(
    name="streaming-stardist",
    title="Streaming tiles - StarDist 2D example",
    parameters={
        "image": sk.sk.Image(dimensionality=[2, 3]),
        "chunk_size": sk.sk.Integer("Chunks", default=64, min=1),
        "overlap": sk.sk.Float("Overlap %", default=0, min=0, max=1),
        "delay": sk.sk.Float("Delay (sec)", default=0.1, min=0),
    },
    samples=["/home/wittwer/data/test_images/C3-bigimg-gray-inv-crop.tif"],
)
def stream_stardist_2d(image, chunk_size, overlap, delay):
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    for tile_meta in sk.generate_nd_tiles(
        pixel_domain=image.shape,
        tile_size_px=chunk_size,
        overlap_percent=overlap,
        delay_sec=delay,
    ):
        image_tile = sk.Image()._get_tile(image, tile_meta)
        segmentation, polys = model.predict_instances(
            normalize(image_tile),
            prob_thresh=0.5,
            nms_thresh=0.4,
            scale=1.0,
        )

        if segmentation.max() > 0:
            yield [
                (
                    segmentation,
                    {"name": "Stardist result"} | tile_meta,
                    "instance_mask",
                )
            ]


if __name__ == "__main__":
    sk.serve(stream_stardist_2d)
