import math
import numpy as np
from logging import getLogger
from typing import Self
from scipy.stats import poisson

from scrs.image.logger import get_log_and_save
from .image import Image


logger = getLogger(__name__)
log_and_save = get_log_and_save(logger)


class Simage(Image):
    """
    Simulation image class, which extends the Image class to include the non-utility functions for simulating images.
    """

    @log_and_save
    def add_stars(
        self, num_stars: int, sat_level: int = 65536, max_sigma: int = 10, allow_sat: bool = False
    ) -> Self:
        """
        Add stars to an image.

        Args:
            num_stars (int): Number of stars to add.
            sat_level (int): Maximum pixel value for a star. Will generate stars up to 10% more than this, but
                will cap them at this value.
            max_sigma (int): The spread of the star. Larger values will make the star larger and more diffuse.
            sig_clip (int): Number of standard deviations to clip the star region at, for performance purposes.
                A sig_clip of 3 will capture 99.7% of the star's light, so anything beyond this starts to become
                negligible.
        """
        img_height, img_width = self._data.shape

        min_star_value: np.uint16 = 0
        max_amp = sat_level * (1.1 if allow_sat else 0.9)

        for _ in range(num_stars):
            # Random star positio np.uint16 = 0n (x0, y0)
            x0 = np.random.randint(0, img_width)
            y0 = np.random.randint(0, img_height)

            # Random amplitude and size (sigma)
            amplitude = np.random.uniform(1, max_amp)
            sigma = np.random.uniform(1, max_sigma)

            # Local region calculations
            min_val = 0.5  # Below this point, will be 0 in uint
            radius = math.ceil(np.sqrt(-2 * sigma**2 * np.log(min_val / amplitude)))

            # Clip the region to the image bounds
            x_min = max(0, x0 - radius)
            x_max = min(img_width, x0 + radius + 1)
            y_min = max(0, y0 - radius)
            y_max = min(img_height, y0 + radius + 1)

            x_local, y_local = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian_star = amplitude * np.exp(
                -((x_local - x0) ** 2 + (y_local - y0) ** 2) / (2 * sigma**2)
            )

            # Add the star to the image
            values = np.maximum(self._data[y_min:y_max, x_min:x_max], gaussian_star)
            if (min_val := np.min(values)) > min_star_value:
                min_star_value = min_val

            self._data[y_min:y_max, x_min:x_max] = values

        # Bring the star values between sat_level and 0
        self._data = np.clip(self._data, 0, sat_level)

        return self

    @log_and_save
    def add_noise(
        self,
        grid_size: tuple[int, int] | None = None,
        bias_values: np.ndarray | None = None,
        noise_stdev: np.ndarray | None = None,
    ) -> Self:
        """
        Adds spatially varying background noise to the data.

        Args:
            grid_size (tuple[int, int], optional): Number of grid cells along each dimension. Default is (4, 2).
            bias_values (np.ndarray, optional): Array specifying the bias for each grid cell. Default is a 2x4 array.
            noise_stdev (np.ndarray, optional): Array specifying the standard deviation of noise for each grid cell. Default is a 2x4 array.
        """
        # Set default values if not provided
        if grid_size is None:
            grid_size = (4, 2)
        if bias_values is None:
            bias_values = np.array([[1000, 1001, 999, 1003], [1002, 998, 1000, 999]])
        if noise_stdev is None:
            noise_stdev = np.array([[2.5, 2.4, 2.5, 2.2], [2.1, 2.4, 2.5, 2.3]])

        # Calculate dimensions for each grid cell
        grid_x, grid_y = grid_size
        cell_height = self._data.shape[0] // grid_x
        cell_width = self._data.shape[1] // grid_y

        # Initialize the bias array
        bias_image = np.zeros_like(self._data)

        # Apply bias and noise to each grid cell
        for x in range(grid_x):
            for y in range(grid_y):
                x_start, x_end = x * cell_height, (x + 1) * cell_height
                y_start, y_end = y * cell_width, (y + 1) * cell_width
                bias_image[x_start:x_end, y_start:y_end] = bias_values[y, x] + np.random.normal(
                    scale=noise_stdev[y, x], size=(cell_height, cell_width)
                )

        # Add Poisson noise and return the result
        self._data = poisson.rvs(self._data) + bias_image
        return self

    @log_and_save
    def add_noise2(
        self,
        poisson_mean: float,
        num_segments: tuple[int, int] = (1, 1),
        bias_values: np.ndarray = None,
        noise_std: np.ndarray = None,
        segment_size: tuple[int, int] = None,
    ):
        """
        Generate an image with Poisson-distributed values and added segment-wise bias.

        Parameters:
        - image_data (np.ndarray): The input image data array for which to generate the Poisson noise.
        - poisson_mean (float): The mean of the Poisson distribution used to generate noise.
        - num_segments (tuple of int): A tuple (num_segments_x, num_segments_y) specifying the number of segments in the x and y dimensions.
        - bias_values (np.ndarray): A 2D array of shape (num_segments_y, num_segments_x) providing the baseline bias values for each segment.
        - noise_std (np.ndarray): A 2D array of shape (num_segments_y, num_segments_x) providing the standard deviation of the noise for each segment.
        - segment_size (tuple of int, optional): A tuple (segment_dim_x, segment_dim_y) specifying the size of each segment. If not provided, segment sizes are calculated based on the image dimensions and num_segments.

        Returns:
        - np.ndarray: The generated image with Poisson noise added to the segment-wise bias.

        Raises:
        - ValueError: If the shapes of bias_values or noise_std do not match the expected (num_segments_y, num_segments_x) dimensions.
        """
        if bias_values is None:
            bias_values = np.array([[1000, 1001, 999, 1003], [1002, 998, 1000, 999]])

        # Dimensions of the image data array
        image_shape = np.shape(self._data)

        # Number of segments along each dimension
        num_segments_x, num_segments_y = num_segments

        # Determine segment dimensions
        if segment_size is None:
            segment_dim_x = image_shape[0] // num_segments_x
            segment_dim_y = image_shape[1] // num_segments_y
        else:
            segment_dim_x, segment_dim_y = segment_size

        # Check if bias_values and noise_std are appropriate
        if np.shape(bias_values) != (num_segments_y, num_segments_x) or np.shape(noise_std) != (
            num_segments_y,
            num_segments_x,
        ):
            raise ValueError(
                "bias_values and noise_std must have shapes matching (num_segments_y, num_segments_x)"
            )

        # Create a meshgrid for the bias calculation
        segment_x, segment_y = np.meshgrid(
            np.arange(num_segments_x), np.arange(num_segments_y), indexing="ij"
        )

        # Compute the bias for each segment
        biases = bias_values[segment_y, segment_x] + np.random.normal(
            scale=noise_std[segment_y, segment_x],
            size=(num_segments_y, num_segments_x, segment_dim_x, segment_dim_y),
        )

        # Initialize the bias array
        bias_array = np.zeros(image_shape)

        # Fill the bias array using the computed biases
        for x in range(num_segments_x):
            for y in range(num_segments_y):
                bias_array[
                    x * segment_dim_x : (x + 1) * segment_dim_x,
                    y * segment_dim_y : (y + 1) * segment_dim_y,
                ] = biases[y, x]

        # Generate and return the final image with Poisson noise added
        return poisson.rvs(poisson_mean) + bias_array
