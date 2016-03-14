import logging
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def calc_num_na(df):
    """ Check number of missing values in data frame
    Keyword Arguments:
    :param df: `pandas.DataFrame`
    """
    return pd.isnull(df).sum().sum()


def load_data(path, sep=',', **kwargs):
    """Load observations for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: keyword arguments passed to `pandas.read_table`
    """
    # TODO: may want to do validation callbacks here
    log.debug('Loading data at %s' %path)
    df = pd.read_table(path, sep=sep, **kwargs)
    return df


# TODO: not robust to large x
def inverse_logit(x):
    """Calculate inverse logit (prob)

    :param x: logit
    """
    return np.exp(x) / (1 + np.exp(x))


def choose_from_multinomial_probs(df):
    """Choose class from data frame of classes probabilities

    index of input data frame is assuemd to contain class names.

    :param df: `pandas.DataFrame` with values i,j corresponding to probability
    of observation i belonging to class j.

    """
    log.debug('Getting name of class with highest probability')
    if df.shape[1] < 2:
        raise ValueError('Data frame must have more than 1 column')

    classes = df.idxmax(axis=1)
    return pd.DataFrame(classes, columns=['class'])


def choose_from_binary_probs(df, name_true='True', name_false='False', threshold=0.5):
    """Choose class from data frame of class probability

    :param df: `pandas.DataFrame` with values i,j corresponding to probability of observation i belonging to class `name_true`
    :param threshold: threshold for assigning observatiosn to class `name_true`
    :param name_true: name of class for positive results
    :param name_false: name of class for negative results

    """
    if df.shape[1] != 1:
        raise ValueError('Data frame must have 1 column')

    if name_false == name_true:
        raise ValueError('Class names for true and false results must differ')

    if threshold < 0 or threshold >1:
        raise ValueError('Threshold must be between 0 and 1')

    log.debug('Assigning class name %s to observations > %s' %(name_true, str(threshold)))
    # TODO: check performance when predictions done row by row
    def apply_threshold(x, name_true=name_true, name_false=name_false,
                        threshold=threshold):
        cls = name_true if x > threshold else name_false
        return cls

    classes = df.ix[:,0].apply(apply_threshold)

    # TODO: a series to dataframe decorator could be convenient
    return pd.DataFrame(classes, columns=['class'])
