from typing import Dict, Tuple

import numpy as np


def one_hot_to_rgb(
    one_hot: np.ndarray,
    classes: Tuple[str, ...],
    palette: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Convert a one-hot encoded tensor into an RGB image.

    Args:
        one_hot (np.ndarray): One-hot encoded tensor of shape (num_classes, H, W).
        classes (tuple): Tuple containing the class names.
        palette (dict): Dictionary mapping class names to RGB values.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    # Initialize the RGB image
    h, w = one_hot.shape[1], one_hot.shape[2]
    rgb_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Iterate over each class and assign the corresponding color
    for idx, class_name in enumerate(classes):
        mask = one_hot[idx] == 1
        rgb_image[mask] = palette[class_name]

    return rgb_image
