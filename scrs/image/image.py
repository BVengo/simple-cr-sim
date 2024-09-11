from logging import getLogger
from pathlib import Path
from typing import Callable, Self
from copy import deepcopy
from uuid import uuid4
from astropy.io import fits

import numpy as np

from scrs.utils import ensure_dir
from .logger import get_log_and_save


logger = getLogger(__name__)
log_and_save = get_log_and_save(logger)



class Image:
    _data: np.ndarray = None  # The image data being manipulated

    _history: list[str] = None  # A list of all operations performed on the image
    _save_path: Path = None  # The path to save the image to

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
        Provides a copy of the history (for read-only purposes).

        Returns:
            list[str]: A list of all operations performed on the image.
        """
        return [h for h in self._history]


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
    def load_history(self, path: Path, step_id: str) -> Self:
        """
        Loads an image from the history.

        Args:
            path (Path): The path to the history folder.
            step_id (str): The ID of the step to load from the history.

        Returns:
            Self: The current instance of the image.
        """
        ensure_dir(path)

        for file in path.iterdir():
            if step_id in file.name:
                return Image().load_fits(file)

        raise FileNotFoundError(f"No file found for with ID {step_id}!.")

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
    def use_history(self, enable: bool = True, path: Path | None = None) -> Self:
        """
        Enable history for this image. All interim operations will be saved
        within the given folder.

        Args:
            enable (bool): Whether to enable history.
            path (Path): The path to save the history to.
        
        Returns:
            Self: The current instance of the image.
        """
        if path is not None:
            if self._save_path is not None:
                logger.warning("Overwriting existing save path.")
            
            ensure_dir(path, create=True)
            self._save_path = path

        if enable:
            if self._history is None:
                self._history = []
            else:
                logger.warning("History is already enabled.")

            if self._save_path is None:
                raise ValueError("Cannot enable history without a save path.")
        else:
            self._history = None
        
        return self        
        
    
    def clear_history(self, delete_files: bool = False) -> Self:
        """
        Clears the history of this image. To save a copy of the data after clearing
        the history, call save_snapshot directly.

        Be warned; if delete_files is called, copies of this will also
        have the copied part of their history cleared. This is to prevent
        accidental dangling references.

        Args:
            delete_files (bool): Whether to delete the files saved in the history.

        Returns:
            Self: The current instance of the image.
        """
        if self._save_path is None:
            return self
        
        if delete_files:
            for h in self._history:
                for file in self._save_path.iterdir():
                    if h in file.name:
                        file.unlink()
                        break
        
        self._history = []

        return self

    def save_snapshot(self, step_name: str = None):
        """
        Takes a copy of the current image and saves it to the history.
        """
        # Check that history is enabled
        if self._history is None or self._data is None:
            return
        
        # Save the image
        step_id = str(uuid4())[0:8]
        hdu = fits.PrimaryHDU(self._data)
        hdu.writeto(self._save_path / f"({step_id}) {step_name}.fits")

        self._history.append(step_id)

        logger.info(f"Saved history [{step_id}] for {step_name}")