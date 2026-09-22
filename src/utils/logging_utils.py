from __future__ import annotations

import logging

_NOISY_LIBRARY_LOGGER_LEVELS: tuple[tuple[str, int], ...] = (
    ("autogluon.common.loaders", logging.ERROR),
    ("autogluon.common.loaders.load_pd", logging.ERROR),
    ("matplotlib", logging.WARNING),
    ("matplotlib.font_manager", logging.WARNING),
)


def raise_noisy_library_log_levels() -> None:
    """Raise noisy third-party loggers to readable defaults.

    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.DEBUG)
        >>> raise_noisy_library_log_levels()
    """
    for logger_name, minimum_level in _NOISY_LIBRARY_LOGGER_LEVELS:
        std_logger = logging.getLogger(logger_name)
        if std_logger.level == logging.NOTSET or std_logger.level < minimum_level:
            std_logger.setLevel(minimum_level)
