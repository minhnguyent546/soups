"""
Use this class with:

```python
from soups.utils.logger import logger, init_logger

init_logger()  # should be called at the start of your script
logger.info('This is an info message')
```
"""

import sys

from loguru import logger


def init_logger(
    level: str = 'DEBUG',
    log_file: str | None = None,
    compact: bool = False,
) -> None:
    if compact:
        fmt = (
            '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{module}:{line}</cyan> - '
            '<level>{message}</level>'
        )
    else:
        fmt = (
            '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}:{function}:{line}</cyan> - '
            '<level>{message}</level>'
        )
    logger.remove()
    logger.add(
        sys.stdout,
        format=fmt,
        level=level,
    )
    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level='DEBUG',
        )
