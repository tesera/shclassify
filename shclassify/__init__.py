import logging
import os

from .core import load_input_data


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILENAME = __name__ + '.log'
_LOG_PATH = os.path.join(_BASE_DIR, _LOG_FILENAME)
_LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


logging.basicConfig(level=logging.DEBUG,
                    format=_LOG_FORMAT,
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename=_LOG_PATH,
                    filemode='w')

log = logging.getLogger(__name__)


__all__ = [load_input_data]
