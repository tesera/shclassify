import pandas as pd

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
    df = pd.read_table(path, sep=sep, **kwargs)
    return df

