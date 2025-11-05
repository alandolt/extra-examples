"""
Example usage of `napari-serverkit` to implement a live stream from a USB camera (e.g. a webcam) into the Napari viewer.
"""

import cv2
import numpy as np
import napari
import imaging_server_kit as sk

from skimage.filters import sobel, threshold_otsu

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
    name="Webcam stream",
    parameters={
        "webcam_idx": sk.Integer(name="Webcam index", min=0),
        "filter": sk.Choice(name="Filter", items=["sobel", "threshold", "none"], default="none")
    },
)
def stream_webcam(webcam_idx, filter):
    camera = VideoCamera(webcam_idx)
    while True:
        frame = camera.get_frame()
        
        if filter == "sobel":
            frame = frame[..., 1]
            frame = sobel(frame)
            yield sk.Image(frame, name="Webcam stream (filtered)")
        elif filter == "threshold":
            frame = frame[..., 1]
            binary = frame > threshold_otsu(frame)
            yield sk.Image(frame, name="Webcam stream (gray)"), sk.Mask(binary, name="Webcam stream (segmented)")
        else:
            yield sk.Image(frame, name="Webcam stream")
        


if __name__ == "__main__":
    import napari
    sk.to_napari(stream_webcam)
    napari.run()
