from pathlib import Path
import numpy as np
import pandas as pd
import trackpy as tp
import imaging_server_kit as sk

from imaging_server_kit.demo.examples import blob_detector_algo


@sk.algorithm(
    name="Tracking (Trackpy)",
    parameters={
        "points": sk.Points(
            name="Points",
            description="The points to track.",
            dimensionality=[2, 3],
        ),
        "search_range": sk.Integer(
            name="Search range",
            description="Search range in pixels.",
            default=30,
            max=100,
        ),
        "memory": sk.Integer(
            name="Memory",
            description="Maximum number of skipped frames for a single track.",
            default=3,
            max=10,
        ),
    },
    project_url="https://soft-matter.github.io/trackpy/",
    description="Fast, Flexible Particle-Tracking Toolkit.",
    tags=["Tracking"],
    samples=[
        {
            "points": pd.read_csv(Path(__file__).parent / "samples" / "detections.csv")[
                ["axis-0", "axis-1", "axis-2"]
            ].values
        },
    ],
)
def trackpy_algo(points, search_range, memory):
    df = pd.DataFrame(
        {
            "frame": points[:, 0],
            "y": points[:, 1],
            "x": points[:, 2],
        }
    )

    linkage_df = tp.link(df, search_range=search_range, memory=memory)

    tracks = linkage_df[["particle", "frame", "y", "x"]].values.astype(float)

    return sk.Tracks(tracks, name="Tracks (trackpy)")


if __name__ == "__main__":
    combined = sk.combine([trackpy_algo, blob_detector_algo], name="Trackpy+LoG")
    sk.serve(combined)
