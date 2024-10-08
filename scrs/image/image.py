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

    @property
    def history(self) -> list[str]:
        """
        Returns the history of operations that have been applied to the image.

        Returns:
            list[str]: The history of operations.
        """
        return self._history

    def get_header(self) -> dict:
        """
        Returns the header of the image.

        Returns:
            dict: The header of the image.
        """
        if self._save_file is None:
            raise ValueError("Cannot get header without a save file.")

        if not self._save_file.exists():
            raise FileNotFoundError(f"File {self._save_file} does not exist.")

        return fits.getheader(self._save_file)

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

        self._save_file = fits_file

        hdr = fits.getheader(fits_file)
        if "HISTORY" in hdr:
            self._history = hdr["HISTORY"]

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

    def add_to_header(self, **kwargs) -> Self:
        """
        Adds a key-value pair to the header of the image.

        Args:
            **kwargs: Key-value pairs to add to the header, as keyword arguments.

        Returns:
            Self: The current instance of the image.
        """
        if self._save_file is None:
            raise ValueError("Cannot add to header without a save file!.")

        if not self._save_file.exists():
            raise FileNotFoundError(
                f"File {self._save_file} does not exist! Make sure that history is enabled."
            )

        hdu_list = fits.open(self._save_file)
        hdr = hdu_list[0].header

        for key, value in kwargs.items():
            hdr.append((key, value))

        hdu_list.writeto(self._save_file, overwrite=True)

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
            if not path.suffix == ".fits":
                raise ValueError("History must be saved to a .fits file.")
            
            if not path.parent.exists():
                raise FileNotFoundError(f"Directory {path.parent} does not exist.")

            if self._save_file is not None:
                logger.warning("Path was already set! Saving to a new file.")

            self._save_file = path

        return self

    def save_snapshot(self, step_name: str) -> None:
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

        self._history.append(history_note)
        logger.info(f"Saved history [{step_id}] for {step_name}")

    def get_snapshot(self, *, query: str | None = None, idx: int | None = None) -> "Image":
        """
        Retrieves a snapshot of the image at the given step or index. Stores it in a new
        Image instance.

        Note that idx is shifted at each step, so idx=0 will return the most recent snapshot and
        idx=-1 will return the oldest snapshot.

        Args:
            query (str): The step ID to retrieve the snapshot for, or the whole history value.

        Returns:
            np.ndarray: The image data at the given step ID.
        """
        if (query is None) == (idx is None):
            raise ValueError("Either query or idx must be provided, but not both.")

        if query is not None:
            # Loop through history to find the corresponding ID
            for i, hist in enumerate(self.history):
                if query in hist:
                    idx = i
                    break

            if idx is None:
                raise ValueError("Could not find the given step in the history.")

        elif -len(self.history) > idx or idx >= len(self.history):
            raise ValueError("Index out of range.")

        hdu_list = fits.open(self._save_file)
        data = hdu_list[idx].data
        return Image(data)

    def get_diff(self, latest_idx: int, oldest_idx: int) -> "Image":
        """
        Retrieves the difference between the image at the latest index and the oldest index.
        i.e. latest_idx - oldest_idx

        Args:
            latest_idx (int): The most recent step ID to start at.
            oldest_idx (int): The oldest step ID to end at.

        Returns:
            np.ndarray: The difference between the current image and the image at the given step ID.
        """
        latest_snap = self.get_snapshot(idx=latest_idx)
        oldest_snap = self.get_snapshot(idx=oldest_idx)

        return Image(latest_snap.data - oldest_snap.data)
