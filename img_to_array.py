from typing import List
import numpy as np
from PIL import Image


def extract_curve_y(
    image_path: str,
    num_points: int = 1000,
    threshold: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract y-values of a dark curve from a grayscale image.

    Args:
        image_path: Path to the image (will be converted to grayscale).
        num_points: Number of samples along the x-axis.
        threshold: Grayscale cutoff (0–255) to detect the curve pixels.
        normalize: If True, scale output to [0, 1] with 0 at bottom.

    Returns:
        A numpy array of shape (num_points,) with the y-value of the curve
        at each sampled x (float). Points with no detected curve will be NaN.
    """
    # load & grayscale
    img = Image.open(image_path).convert("L")
    arr = np.array(img)  # shape (H, W)
    H, W = arr.shape

    # binary mask of where the curve is (dark pixels)
    mask = arr < threshold

    # sample x positions in pixel space
    xs = np.linspace(0, W - 1, num_points)
    ys: List[float] = []

    for x in xs:
        col = mask[:, int(round(x))]
        idx = np.where(col)[0]
        if idx.size:
            # take the average y of all detected pixels in this column
            y_px = float(idx.mean())
            # invert so y=0 at bottom
            y_val = (H - 1 - y_px) if normalize else y_px
        else:
            y_val = float("nan")
        ys.append(y_val)

    y_arr = np.array(ys, dtype=float)
    if normalize:
        y_arr /= float(H - 1)

    return y_arr


# --- usage ---
if __name__ == "__main__":
    y_values = extract_curve_y(
        "img/climb_curve.png",
        num_points=25,
    )
    print(y_values.shape)  # -> (1000,)
    # print(y_values)       # uncomment to dump all 1,000 values
