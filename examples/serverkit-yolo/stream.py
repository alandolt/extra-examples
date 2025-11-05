import numpy as np
import time
import cv2
from ultralytics import YOLO

import imaging_server_kit as sk

from main import postprocess_results

class VideoCamera:
    def __init__(self, video_idx: int):
        self.video = cv2.VideoCapture(video_idx)

    def __del__(self):
        self.video.release()

    def get_frame(self) -> np.ndarray:
        success, image = self.video.read()
        if not success:
            raise RuntimeError("Failed to capture frame from camera")

        image = image[..., ::-1]  # BGR => RGB

        return image


@sk.algorithm(
    name="Yolo stream",
    parameters={
        "webcam_idx": sk.Integer(name="Webcam index", min=0),
        "device": sk.Choice(
            name="Device",
            description="Torch device for inference.",
            default="cpu",
            items=["cpu", "cuda", "mps"],
        ),
    },
)
def yolo_detect_live(
    webcam_idx: int,
    device: str,
):
    # model = YOLO("yolo11n.pt")
    model = YOLO("/home/wittwer/code/yolo-workshop/models/quinoa_500ep/train/weights/best.pt")
    camera = VideoCamera(webcam_idx)
    while True:
        frame = camera.get_frame()

        results = model(source=frame, device=device)

        probabilities = results[0].boxes.conf.cpu().numpy()
        n_detections = len(probabilities)
        if n_detections == 0:
            yield sk.String("No detections.")
            continue
        
        boxes, classes_indeces = postprocess_results(results)
        classes = [str(model.names.get(class_index)) for class_index in classes_indeces]
        unique_classes = np.unique(np.array(classes))

        yield sk.String(f"Detections: {n_detections} (Classes: {unique_classes})")
        
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
        
        # When serving, adding a little delay can help to stabilize the frame rate (TODO: investigate)
        time.sleep(0.1)

        yield sk.Image(frame, name="Camera stream", meta={"contrast_limits": [0, 255]}), sk.Boxes(boxes, meta=boxes_params)


if __name__ == "__main__":
    sk.serve(yolo_detect_live)
