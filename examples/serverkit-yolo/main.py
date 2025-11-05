from pathlib import Path
import numpy as np
from ultralytics import YOLO
import imaging_server_kit as sk


def postprocess_results(results):
    box_results = results[0].boxes.xyxy.cpu().numpy().reshape((-1, 2, 2))
    classes_indeces = results[0].boxes.cls.cpu().numpy()
    boxes = []
    for [(x0, y0), (x1, y1)] in box_results:
        boxes.append(
            [
                [x0, y0],
                [x0, y1],
                [x1, y1],
                [x1, y0],
            ]
        )
    boxes = np.array(boxes)
    boxes = boxes[..., ::-1]  # Invert X-Y
    return boxes, classes_indeces

@sk.algorithm(
    description="Real-time object detection with YOLO (Ultralytics implementation).",
    name="Object detection (YOLO)",
    tags=["Bounding box", "Deep learning", "YOLO"],
    parameters={
        "image": sk.Image(name="Image (2D, RGB)", description="Input image."),
        "iou": sk.Float(
            name="IoU",
            description="Intersection over union threshold.",
            default=0.5,
            min=0,
            max=1.0,
            step=0.1,
        ),
        "conf": sk.Float(
            name="Conf.",
            description="Confidence threshold for detection.",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
        ),
        "device": sk.Choice(
            name="Device",
            description="Torch device for inference.",
            items=["cpu", "cuda", "mps"],
            default="cpu",
        ),
    },
    samples=[{"image": Path(__file__).parent / "samples" / "giraffs.png"}],
)
def yolo_detect_algo(
    image: np.ndarray,
    iou: float,
    conf: float,
    device: str,
):
    """Run a pretrained YOLO detector model."""
    model = YOLO("yolo11n.pt")

    if image.shape[2] == 4:  # RGBA to RGB
        image = image[..., :3]

    results = model(
        source=image,
        conf=conf,
        iou=iou,
        device=device,
    )

    probabilities = results[0].boxes.conf.cpu().numpy()
    n_detections = len(probabilities)
    if n_detections == 0:
        return sk.Notification("No detections.")

    boxes, classes_indeces = postprocess_results(results)
    classes = [str(model.names.get(class_index)) for class_index in classes_indeces]
    unique_classes = np.unique(np.array(classes))

    boxes_params = {
        "name": "YOLO detections",
        "face_color": "transparent",
        "opacity": 1.0,
        "edge_width": 1,
        "edge_color": "class",
        "features": {
            "probability": probabilities,
            "class": classes,
        },
    }

    return sk.Boxes(boxes, meta=boxes_params), sk.String(f"Detections: {n_detections} (Classes: {unique_classes})")


if __name__ == "__main__":
    sk.serve(yolo_detect_algo)
