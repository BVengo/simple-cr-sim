from logging import getLogger
from pathlib import Path
from typing import Callable, Self
from copy import deepcopy
from uuid import uuid4
from astropy.io import fits

import numpy as np

from .logger import get_log_and_save


logger = getLogger(__name__)
log_and_save = get_log_and_save(logger)


class Image:
    _data: np.ndarray = None  # The image data being manipulated
    _save_history: bool = False  # Whether to save history
    _save_file: Path = None  # The path to save the image to
    _history: list[str] | None = None  # The history of operations

    def __init__(self, data: np.ndarray = None):
        if data is not None:
            self.set_data(data)

        self._history = []

    @property
    def data(self) -> np.ndarray:
        """
        Provides a deep copy of the data to ensure the Image data is not modified,
        so that history can be retained.

        If you want to applying operations on the Image data, there are two options available:
        1. Use this property to get a deep copy, and then use `set_data` to update the image with your changes.
        2. Use `apply` to apply operations to the data without requiring a deepcopy.

        Returns:
            np.ndarray: A deep copy of the image data.
        """
        return deepcopy(self._data)

    @log_and_save
    def load_fits(self, fits_file: Path) -> Self:
        """
        Loads a fits file into the image data.

        Args:
            fits_file (Path): The path to the fits file to load.

        Returns:
            Self: The current instance of the image.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not fits_file.exists():
            raise FileNotFoundError(f"File {fits_file} does not exist")

        if self._data is not None:
            logger.warning("Call to `load_fits` is overwriting existing data!")

        self._data = fits.getdata(fits_file)

        return self

    @log_and_save
    def set_data(self, data: np.ndarray) -> Self:
        """
        Sets the image data to the given data.

        Args:
            data (np.ndarray): The data to set the image to.

        Returns:
            Self: The current instance of the image.
        """
        if self._data is not None:
            logger.warning("Call to `set_data` is overwriting existing data!")

        self._data = data

        return self

    @log_and_save
    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> Self:
        """
        Applies a function to the image data. The function must take in the image data
        and return the modified image data.

        Args:
            func: The function to apply to the image data.

        Returns:
            Self: The current instance of the image.
        """
        if self._data is None:
            raise ValueError("Cannot apply operations to empty data.")

        self._data = func(self._data)

        return self

    @log_and_save  # Required to save anything up to this point
    def enable_history(self, path: Path | None = None, enable: bool = True) -> Self:
        """
        Enable history for this image. All operations from this point will be saved
        within the given fits file.

        Args:
            enable (bool): Whether to enable history.
            path (Path): The file path to save the history to.

        Returns:
            Self: The current instance of the image.
        """
        self._save_history = enable

        if enable and path is None and self._save_file is None:
            raise ValueError("Cannot enable history without a save path.")

        if path:
            if self._save_file is not None:
                logger.warning("Path was already set! Saving to a new file.")

            self._save_file = path

        return self

    def save_snapshot(self, step_name: str = None):
        """
        Takes a copy of the current image and saves it to the history.
        """
        # Check that history is enabled
        if not self._save_history or self._data is None:
            return

        step_id = str(uuid4())
        history_note = f"[{step_id}] {step_name}"

        if self._save_file.exists() and len(self._history) > 0:
            # There is history, so this must be part of the pipeline. Append to the existing file.
            hdu_list = fits.open(self._save_file)

            hdr = hdu_list[0].header
            hdr.append(("HISTORY", history_note))

            # Set the new data as the primary HDU. Shift the previous primary HDU to the second HDU.
            new_hdu = fits.PrimaryHDU(self._data, header=hdr)
            prev_hdu = fits.ImageHDU(hdu_list.pop(0).data)

            hdu_list = fits.HDUList([new_hdu, prev_hdu] + hdu_list)

        else:
            # Save the image as a new file
            hdr = fits.Header({"HISTORY": history_note})
            hdu = fits.PrimaryHDU(self._data, header=hdr)
            hdu_list = fits.HDUList([hdu])

        hdu_list.writeto(self._save_file, overwrite=True)

        self._history.append(step_id)
        logger.info(f"Saved history [{step_id}] for {step_name}")
