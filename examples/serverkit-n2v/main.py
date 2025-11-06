from pathlib import Path
import numpy as np
import shutil
from careamics import CAREamist
from careamics.config import create_n2v_configuration
import imaging_server_kit as sk


@sk.algorithm(
    parameters={
        "image": sk.Image(
            description="Input image (2D, 3D). If 2D, myst be grayscale (not RGB). If 3D, the image is considered a 2D series."
        ),
        "epochs": sk.Integer(
            name="Epochs",
            description="Number of epochs for training",
            default=10,
            min=1,
            max=1000,
        ),
        "patch_size": sk.Integer(
            name="Patch size",
            description="Square patch size in pixels. Must be a power of two (16, 32, 64...).",
            default=16,
            min=3,
            max=1024,
        ),
        "batch_size": sk.Integer(
            name="Batch size",
            description="Batch size for training",
            default=4,
            min=1,
            max=1024,
        ),
    },
    samples=[{"image": Path(__file__).parent / "samples" / "blobs-noisy.tif", "patch_size": 32}],
)
def n2v_algo(image, epochs, patch_size, batch_size):
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

    return sk.Image(prediction, name=f"Denoised ({epochs} epochs)")


if __name__ == "__main__":
    sk.serve(n2v_algo)
