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

        max_amp = sat_level * (1.1 if allow_sat else 0.9)

        # Random star attributes.
        x_positions = np.random.randint(0, img_width, num_stars)
        y_positions = np.random.randint(0, img_height, num_stars)
        amplitudes = np.random.uniform(1, max_amp, num_stars)
        sigmas = np.random.uniform(1, max_sigma, num_stars)
        radii = np.ceil(np.sqrt(-2 * sigmas**2 * np.log(0.5 / amplitudes)))

        # Clip the region to the image bounds. Set all as integers.
        x_mins = np.maximum(0, x_positions - radii).astype(int)
        x_maxs = np.minimum(img_width, x_positions + radii + 1).astype(int)
        y_mins = np.maximum(0, y_positions - radii).astype(int)
        y_maxs = np.minimum(img_height, y_positions + radii + 1).astype(int)

        for i in range(num_stars):
            x0, y0, amplitude, sigma = x_positions[i], y_positions[i], amplitudes[i], sigmas[i]
            x_min, x_max, y_min, y_max = x_mins[i], x_maxs[i], y_mins[i], y_maxs[i]

            x_local, y_local = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian_star = amplitude * np.exp(
                -((x_local - x0) ** 2 + (y_local - y0) ** 2) / (2 * sigma**2)
            )

            # Add the star to the image. If the current value is greater than the star value,
            # keep it.
            region = slice(y_min, y_max), slice(x_min, x_max)
            self._data[region] = np.maximum(
                self._data[region], gaussian_star
            )

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
