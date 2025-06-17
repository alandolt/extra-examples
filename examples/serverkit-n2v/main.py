from typing import List
from pathlib import Path
import numpy as np
import uvicorn

from imaging_server_kit import algorithm_server, ImageUI, IntUI

import shutil
from careamics import CAREamist
from careamics.config import create_n2v_configuration


@algorithm_server(
    algorithm_name="n2v",
    parameters={
        "image": ImageUI(
            description="Input image (2D, 3D). If 2D, myst be grayscale (not RGB). If 3D, the image is considered a 2D series."
        ),
        "epochs": IntUI(
            default=10,
            title="Epochs",
            description="Number of epochs for training",
            min=1,
            max=1000,
        ),
        "patch_size": IntUI(
            default=16,
            title="Patch size",
            description="Square patch size in pixels. Must be a power of two (16, 32, 64...).",
            min=3,
            max=1024,
        ),
        "batch_size": IntUI(
            default=4,
            title="Batch size",
            description="Batch size for training",
            min=1,
            max=1024,
        ),
    },
    sample_images=[Path(__file__) / "sample_images" / "blobs-noisy.tif"],
    metadata_file="metadata.yaml",
)
def n2v_server(
    image: np.ndarray,
    epochs: int,
    patch_size: int,
    batch_size: int,
) -> List[tuple]:
    """Runs the algorithm."""
    image_ndim = len(image.shape)

    if image.ndim == 2:
        # Consider it a single image XY type
        config = create_n2v_configuration(
            experiment_name="foo",
            data_type="array",
            axes="YX",
            patch_size=(patch_size, patch_size),
            batch_size=batch_size,
            num_epochs=epochs,
        )

    elif image.ndim == 3:
        # Consider it a Sample-YX type
        config = create_n2v_configuration(
            experiment_name="foo",
            data_type="array",
            axes="SYX",
            patch_size=(patch_size, patch_size),
            batch_size=batch_size,
            num_epochs=epochs,
        )

    careamist = CAREamist(source=config)

    careamist.train(train_source=image)

    prediction = careamist.predict(source=image)
    prediction = np.squeeze(np.array(prediction))

    # Not sure how to prevent any logging and checkpoints saving, so for now we just remove the directories created by n2v...
    shutil.rmtree(Path(__file__).parent / "csv_logs")
    shutil.rmtree(Path(__file__).parent / "checkpoints")

    return [(prediction, {"name": "Denoised"}, "image")]


if __name__ == "__main__":
    uvicorn.run(n2v_server, host="0.0.0.0", port=8000)
