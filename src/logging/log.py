import logging
import sys

def get_logger(name, set_level=logging.INFO):
    """Creates a logger object.
    
    Args:
        name: The name of the logger. Must be a string.
        set_level: The level to set the logger to. See python documentation
        to see details info. Default is set to logging.INFO.

        Possible levels:
        debug, info, warning, error, critical

    Returns:
        A logger object.

    Raises:
        TypeError if name is not string.
    """

    if not isinstance(name, str):
        raise TypeError(f"Input name must be string. Instead its datatype is {type(name)}")

    logger = logging.getLogger(name)
    logger.setLevel(set_level)

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger