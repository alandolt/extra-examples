from pathlib import Path
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_napari_tracks
import imaging_server_kit as sk
import skimage.io


@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "mask": sk.Mask(),
        "mode": sk.Choice(
            name="Mode",
            description="Tracking mode.",
            items=["greedy", "greedy_nodiv"],
            default="greedy",
        ),
    },
    samples=[
        {
            "image": Path(__file__).parent / "samples" / "trpL_150310-11_img.tif",
            "mask": skimage.io.imread(
                Path(__file__).parent / "samples" / "trpL_150310-11_mask.tif"
            ),
        },
    ],
)
def trackastra_algo(image, mask, mode):
    model = Trackastra.from_pretrained("general_2d", device="cpu") # cpu for now, to avoid CUDA errors
    graph, masks_tracked = model.track(image, mask, mode)
    tracks, *_ = graph_to_napari_tracks(graph)
    return sk.Tracks(tracks)


if __name__ == "__main__":
    sk.serve(trackastra_algo)
