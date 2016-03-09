import pandas as pd


def load_input_data(path, sep=',', **kwargs):
    """Load input data for SLS HRIS LCC Prediction

    Keyword Arguments:
    path     -- path to input data, string or URL (e.g. http, s3)
    sep      -- input data field separator
    **kwargs -- keyword arguments passed to `pandas.read_table`
    """
    df = pd.read_table(path, sep=sep, **kwargs)
    return df

