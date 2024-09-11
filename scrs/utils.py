from pathlib import Path


def ensure_dir(path: Path, create: bool = False, is_empty: bool = False):
    """
    A simple utility function to ensure a directory exists, or
    create it if it doesn't.

    Args:
        path (Path): The path to the directory to ensure exists.

    Raises:
        ValueError: If the path exists but is not a directory.
        ValueError: If is_empty = True, but the given directory contains files
    """
    if not path.exists():
        if create:
            path.mkdir()
        else:
            raise ValueError(f"{path} does not exist")
    elif not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    elif is_empty:
        # Ensure the existing directory is empty
        if len(list(path.iterdir())) > 0:
            raise ValueError(f"{path} is not empty")
