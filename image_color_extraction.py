"""Extracts and clusters dominant colors from an image for analysis and visualization."""
"""
Image Color Extraction Tool

Author: Ali Bahrami
GitHub: https://github.com/alibahrami2001
Date: 11/3/2024
"""
import io
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def get_basic_colors(each_channels_possible_values: int = 8) -> np.ndarray:
    basic_colors = []
    for i in np.linspace(0, 1, each_channels_possible_values).tolist():
        for j in np.linspace(0, 1, each_channels_possible_values).tolist():
            for k in np.linspace(0, 1, each_channels_possible_values).tolist():
                basic_colors.append([i, j, k])
    basic_colors = np.array(basic_colors, dtype=np.float32)

    # sort colors by their luminance
    basic_colors_luminance = (
        0.299 * basic_colors[:, 0]
        + 0.587 * basic_colors[:, 1]
        + 0.114 * basic_colors[:, 2]
    )
    basic_colors = basic_colors[np.argsort(basic_colors_luminance)]
    return basic_colors


def extract_image_colors(
    image: Image.Image,
    basic_colors: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    # Get image pixel values
    pixels = np.asarray(image).reshape(-1, 3) / 255.0

    # Assign each pixel to a basic color
    assigned_colors = []
    for i in range(0, len(pixels), batch_size):
        assigned_colors.append(
            cdist(pixels[i : i + batch_size], basic_colors, "euclidean").argmin(axis=1)
        )
    assigned_colors = np.concatenate(assigned_colors)

    # extract colors
    extracted_colors = []
    for basic_color_id in range(len(basic_colors)):
        mask = assigned_colors == basic_color_id
        counts = mask.sum()
        if counts > 0:
            extracted_colors.append(pixels[mask].mean(axis=0))

    extracted_colors = np.array(extracted_colors)

    return extracted_colors


def cluster_extracted_colors(
    extracted_colors: np.ndarray,
    n_clusters: int = 8,
    random_state: int | None = 42,
):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
        extracted_colors
    )
    colors = kmeans.cluster_centers_

    # sort colors by their luminance
    color_luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    colors = colors[np.argsort(color_luminance)]
    return colors


class Color(NamedTuple):
    red: int
    green: int
    blue: int


class ImageColorExtractor:
    """Extracts and generates a set of basic colors for image color analysis.

    Args:
        each_channels_possible_values (int, optional): The number of discrete values for each color channel (red, green, blue). 
        This parameter determines the granularity of color representation. Defaults to 8.
    """

    def __init__(self, each_channels_possible_values: int = 8):
        self.basic_colors = get_basic_colors(each_channels_possible_values)

    def __call__(
        self,
        image: Image.Image,
        batch_size: int = 4096,
        max_num_colors: int | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Extracts colors from the provided image.

        Args:
            image (Image.Image): The image from which to extract colors.
            batch_size (int, optional): The number of pixels to process in each batch during the distance calculation. 
            Defaults to 4096.
            max_num_colors (int | None, optional): The maximum number of colors to extract. If set to None, all extracted 
            colors will be returned. If specified, the colors will be clustered using KMeans to reduce the output to 
            this number of colors.
            random_state (int | None, optional): Seed for the random number generator used in KMeans clustering. 
            Defaults to None.

        Returns:
            np.ndarray: An array of extracted colors represented in the range [0, 1].
        """
        extracted_colors = extract_image_colors(image, self.basic_colors, batch_size)

        if max_num_colors is not None:
            num_extracted_colors = len(extracted_colors)
            if num_extracted_colors <= max_num_colors:
                result = extracted_colors
            else:
                result = cluster_extracted_colors(
                    extracted_colors,
                    max_num_colors,
                    random_state=random_state,
                )
        else:
            result = extracted_colors

        return result


def visualize_colors(extracted_colors: np.ndarray) -> Image.Image:
    plt.imshow(extracted_colors.reshape(1, -1, 3), aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer)
    plt.close()
    buffer.seek(0)
    colors = Image.open(buffer)
    return colors