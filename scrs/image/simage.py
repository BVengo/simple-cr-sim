import numpy as np
from logging import getLogger
from typing import Self

from scipy.ndimage import gaussian_filter

from scrs.image.logger import get_log_and_save
from .image import Image
from ..cosmics import simulate_crs

logger = getLogger(__name__)
log_and_save = get_log_and_save(logger)


class Simage(Image):
    """
    Simulation image class, which extends the Image class to include the non-utility functions for simulating images.
    """

    _rng: np.random.Generator

    def __init__(self):
        super().__init__()
        self.set_rng()

    def set_rng(self, seed: int | None = None) -> Self:
        """
        Set the random number generator for the image.

        Args:
            seed (int, optional): Seed for the random number generator. If None, the seed is generated randomly.

        Returns:
            Self: The current instance of the class.
        """
        self._rng = np.random.default_rng(seed)
        return self

    @log_and_save
    def add_stars(
        self, num_stars: int, sat_level: int = 65535, max_sigma: int = 10, allow_sat: bool = False
    ) -> Self:
        """
        Add stars to an image.

        Args:
            num_stars (int): Number of stars to add.
            sat_level (int): Maximum pixel value for a star. Will generate stars up to 10% more than this, but
                will cap them at this value.
            max_sigma (int): The spread of the star. Larger values will make the star larger and more diffuse.
            allow_sat (bool): If saturation of stars should be allowed. If yes, max saturation is set to
                sat_level * 1.1, otherwise it is sat_level * 0.9.

        Returns:
            Self: The current instance of the class.
        """
        img_height, img_width = self._data.shape

        max_amp = sat_level * (1.1 if allow_sat else 0.9)

        # Random star attributes.
        x_positions = (self._rng.random(num_stars) * img_width).astype(np.uint16)
        y_positions = (self._rng.random(num_stars) * img_height).astype(np.uint16)
        amplitudes = self._rng.uniform(1, max_amp, num_stars)
        sigmas = self._rng.uniform(1, max_sigma, num_stars)
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
            self._data[region] = np.maximum(self._data[region], gaussian_star)

        # Bring the star values between sat_level and 0
        self._data = np.clip(self._data, 0, sat_level)

        return self

    # @log_and_save  # `apply` already logs the operation
    def add_cosmics(
        self,
        time: float = 1000,
        flux: float = 8,
        area: float = 16.8,
        conversion_factor: float = 0.5,
        pixel_size: float = 10,
        pixel_depth: float = 5,
    ) -> Self:
        """
        Add cosmic ray strikes to the image.

        Args:
            time (float): Exposure time in seconds.
            flux (float): Cosmic ray flux in strikes per second per square meter.
            area (float): Area of the detector in square meters.
            conversion_factor (float): Conversion factor from strikes to electrons.
            pixel_size (float): Pixel size in micrometers.
            pixel_depth (float): Pixel depth in micrometers.

        Returns:
            Self: The current instance of the class.
        """
        return self.apply(
            lambda data: simulate_crs(
                self._data,
                time=time,
                flux=flux,
                area=area,
                conversion_factor=conversion_factor,
                pixel_size=pixel_size,
                pixel_depth=pixel_depth,
                rng=self._rng,
            )
        )

    @log_and_save
    def add_noise(
        self,
        sigma: float = 2.3,
        peak: float = 1,
        grid_size: tuple[int, int] = (4, 4),
        bias_range: tuple[int, int] = (995, 1005),
        stdev_range: tuple[float, float] = (2.0, 2.5),
    ) -> Self:
        """
        Adds spatially varying background noise to the data.

        Args:
            sigma (float, optional): Standard deviation of the Gaussian kernel used for blurring the image. Default is 2.3.
            peak (float, optional): Peak value of the blurred image. Default is 65535.
            grid_size (tuple[int, int], optional): Number of grid cells along each dimension. Default is (4, 4).
            bias_range (tuple[int, int], optional): Range of bias values for each grid cell. Default is (995, 1005).
            stdev_range (tuple[float, float], optional): Range of standard deviation values for each grid cell. Default is (2.0, 2.5).

        Returns:
            Self: The current instance of the class.
        """
        # Blur and scale(?)
        blurred_data: np.ndarray = gaussian_filter(self._data, sigma=sigma)
        # blurred_data = peak * blurred_data / blurred_data.max()

        # Set bias values for each grid cell
        bias_values = self._rng.integers(*bias_range, size=grid_size)
        noise_stdev = self._rng.uniform(*stdev_range, size=grid_size)

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
                bias_image[x_start:x_end, y_start:y_end] = bias_values[y, x] + self._rng.normal(
                    scale=noise_stdev[y, x], size=(cell_height, cell_width)
                )

        # Add Poisson noise and return the result
        self._data = self._rng.poisson(blurred_data) #  + bias_image
        return self
