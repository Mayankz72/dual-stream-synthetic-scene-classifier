import cv2
import numpy as np
from PIL import Image


class EdgeTransform:
    """Converts an RGB PIL image into a 3-channel, material-invariant edge map.

    Pipeline: grayscale -> bilateral filter (suppresses specular highlights on
    metallic objects while preserving true boundaries) -> Canny / Sobel /
    Laplacian, each stacked as one output channel. This lets the edge stream
    learn object geometry (shape, size, boundaries) independent of color and
    material appearance.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"))
        arr = cv2.bilateralFilter(arr, 9, 150, 150)

        canny = cv2.Canny(arr, 30, 100).astype(np.float32) / 255.0

        sx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sx**2 + sy**2)
        sobel = (
            np.clip(sobel / sobel.max(), 0, 1).astype(np.float32)
            if sobel.max() > 0
            else sobel.astype(np.float32)
        )

        lap = np.abs(cv2.Laplacian(arr, cv2.CV_64F, ksize=3))
        lap = (
            np.clip(lap / lap.max(), 0, 1).astype(np.float32)
            if lap.max() > 0
            else lap.astype(np.float32)
        )

        edge_img = np.stack([canny, sobel, lap], axis=2)
        return Image.fromarray((edge_img * 255).astype(np.uint8))
