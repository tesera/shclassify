import logging
import os

from .core import (load_observations, load_model,
                   generate_fake_observations, calculate_prob)
from .config import BASE_DIR, DATA_DIR, MODEL_FILES


LOG_FILENAME = __name__ + '.log'
LOG_PATH = os.path.join(BASE_DIR, LOG_FILENAME)
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename=LOG_PATH,
                    filemode='w')

log = logging.getLogger(__name__)


__all__ = [DATA_DIR, MODEL_FILES, load_observations, load_model,
           generate_fake_observations, calculate_prob]
