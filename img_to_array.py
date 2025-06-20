from typing import List
import numpy as np
from PIL import Image


def img_to_array(
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


def percentiles(y_values: np.ndarray, percentiles: list) -> dict:
    return {p: float(np.percentile(y_values, p)) for p in percentiles}


# --- usage ---
if __name__ == "__main__":
    y_values = img_to_array(
        "img/climb.png",
        num_points=25,
    )
    print(y_values.shape)  # -> (1000,)
    print(y_values)  # uncomment to dump all 1,000 values

    # use matplotlib to visualize
    import matplotlib.pyplot as plt

    plt.plot(y_values)
    plt.xlabel("x position (sampled)")
    plt.ylabel("y value (normalized)")
    plt.title("Extracted Curve Y-Values")
    # plt.show()

    # smooth the curve into a new array and plot that too
    from scipy.ndimage import gaussian_filter1d

    smoothed_y = gaussian_filter1d(y_values, sigma=2)
    plt.plot(smoothed_y, label="Smoothed Curve", color="orange")
    plt.xlabel("x position (sampled)")
    plt.ylabel("y value (normalized)")
    plt.title("Smoothed Curve Y-Values")
    plt.legend()
    print("Done extracting and plotting curve y-values.")

    # Find the 10th, 20th, 80th, and 90th percentile y-values from the smoothed_y
    percentile_ranges = [10, 20, 80, 90]
    percentile_values = percentiles(smoothed_y, percentile_ranges)
