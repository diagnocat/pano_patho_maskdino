import cv2
import numpy as np
import scipy.ndimage


def find_working_field(
    image: np.ndarray,
    intensity_thresh: float = 0.05,
    bar_size_percent_thresh: float = 0.33,
    bin_thresh: int = 25,
) -> tuple[int, int, int, int]:
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # black bars
    yminb, ymaxb, xminb, xmaxb = find_mono_bars(
        image, intensity_thresh, bar_size_percent_thresh, bin_thresh
    )
    # white bars
    yminw, ymaxw, xminw, xmaxw = find_mono_bars(
        255 - image, intensity_thresh, bar_size_percent_thresh, bin_thresh
    )
    ymin = max(yminb, yminw)
    ymax = min(ymaxb, ymaxw)
    xmin = max(xminb, xminw)
    xmax = min(xmaxb, xmaxw)

    return ymin, ymax, xmin, xmax


def find_mono_bars(
    image: np.ndarray,
    intensity_thresh: float,
    bar_size_percent_thresh: float,
    bin_thresh: int,
) -> tuple[int, int, int, int]:

    _, binary_image = cv2.threshold(image, bin_thresh, 255, cv2.THRESH_BINARY)
    binary_image = scipy.ndimage.binary_erosion(binary_image, iterations=10)

    height, width = binary_image.shape[0], binary_image.shape[1]
    ymin, ymax, xmin, xmax = 0, height - 1, 0, width - 1

    height_bar_size = height * bar_size_percent_thresh
    width_bar_size = width * bar_size_percent_thresh
    while binary_image[ymin, :].mean() < intensity_thresh and ymin < height_bar_size:
        ymin += 1

    while (
        binary_image[ymax, :].mean() < intensity_thresh
        and (height - ymax) < height_bar_size
    ):
        ymax -= 1

    while binary_image[:, xmin].mean() < intensity_thresh and xmin < width_bar_size:
        xmin += 1

    while (
        binary_image[:, xmax].mean() < intensity_thresh
        and (width - xmax) < width_bar_size
    ):
        xmax -= 1

    return ymin, ymax + 1, xmin, xmax + 1
