from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from scrs.image.image import Image
    from logging import Logger


def get_log_and_save(logger: "Logger") -> callable:
    """
    Returns a decorator to log and save the history of an image.
    """

    def decorator(func):
        """
        A decorator to return self from a method.
        """

        def wrapper(instance: "Image", *args, **kwargs) -> Any:
            ret = func(instance, *args, **kwargs)
            logger.info(f"Ran {func.__name__} on image.")

            instance.save_snapshot(func.__name__)
            return ret

        return wrapper

    return decorator
