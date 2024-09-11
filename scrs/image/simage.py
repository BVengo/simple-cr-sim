import math
import numpy as np
from logging import getLogger
from typing import Self

from scrs.image.logger import get_log_and_save
from .image import Image


logger = getLogger(__name__)
log_and_save = get_log_and_save(logger)


class Simage(Image):
    """
    Simulation image class, which extends the Image class to include the non-utility functions for simulating images.
    """
    @log_and_save
    def add_stars(self, num_stars: int, sat_level: int = 65536, max_sigma: int = 10, sig_clip: int = 3) -> Self:
        """
        Add stars to an image.

        Args:
            image (np.ndarray): Image to add stars to.
            num_stars (int): Number of stars to add.
            sat_level (int): Maximum pixel value for a star. Will generate stars up to 10% more than this, but
                will cap them at this value.
            max_sigma (int): The spread of the star. Larger values will make the star larger and more diffuse.
            sig_clip (int): Number of standard deviations to clip the star region at, for performance purposes.
                A sig_clip of 3 will capture 99.7% of the star's light, so anything beyond this starts to become
                negligible.
        """
        img_height, img_width = self._data.shape

        for _ in range(num_stars):
            # Random star position (x0, y0)
            x0 = np.random.randint(0, img_width)
            y0 = np.random.randint(0, img_height)

            # Random amplitude and size (sigma)
            amplitude = np.random.uniform(1, sat_level * 1.1)  # Allow stars to saturate
            sigma = np.random.uniform(1, max_sigma)

            # Local region calculations
            size = math.ceil(sig_clip * sigma)
            x_min = max(0, x0 - size)
            x_max = min(img_width, x0 + size)
            y_min = max(0, y0 - size)
            y_max = min(img_height, y0 + size)

            # Create meshgrid for the local area around the star
            x_local, y_local = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian_star = amplitude * np.exp(-((x_local - x0)**2 + (y_local - y0)**2) / (2 * sigma**2))

            # Add the star to the image
            self._data[y_min:y_max, x_min:x_max] += gaussian_star

            # TODO: Add diffractive spikes

            # Cap over-saturated stars. Don't worry about spillage; that's the same as having a star with a lower sigma being saturated.
            # This is done directly to the image, otherwise existing noise may cause the values to be too high.
            self._data[y_min:y_max, x_min:x_max] = np.clip(self._data[y_min:y_max, x_min:x_max], 0, sat_level)

        return self

    @log_and_save
    def add_readout(self, readout_noise: float = 0.0, readout_offset: float = 0.0) -> Self:
        """
        Add readout noise to an image.

        Args:
            image (np.ndarray): Image to add readout noise to.
            readout_noise (float): Standard deviation of the readout noise.
            readout_offset (float): Offset of the readout noise.
        """
        if readout_noise == 0.0:
            return

        img_height, img_width = self._data.shape
        readout = np.random.normal(readout_offset, readout_noise, (img_height, img_width))
        self._data += readout
        
        return self
