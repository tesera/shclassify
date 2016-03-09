import logging
import os


_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
_LOG_FILENAME = __name__ + '.log'
_LOG_PATH = os.path.join(_BASE_DIR, _LOG_FILENAME)

_LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG,
                    format=_LOG_FORMAT,
                    datefmt='%m-%d %H:%M',
                    filename=_LOG_PATH,
                    filemode='a')


def has_legs():
    return False
