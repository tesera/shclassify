import logging
import os

from .core import load_observations, load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = __name__ + '.log'
LOG_PATH = os.path.join(BASE_DIR, LOG_FILENAME)
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
DATA_DIR = os.path.join(BASE_DIR, 'data')

logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename=LOG_PATH,
                    filemode='w')

log = logging.getLogger(__name__)


__all__ = [DATA_DIR, load_observations, load_model]
