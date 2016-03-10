from .utils import calc_num_na, load_data


def load_model(path, sep=',', **kwargs):
    """Load a model for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: keyword arguments passed to `utils.load_data`
    """
    model = load_data(path, sep=sep, **kwargs)

    missing = calc_num_na(model)
    if missing > 0:
        raise ValueError(('Model has %s missing coefficients.' %missing))

    return model


def load_observations(path, sep=',', **kwargs):
    """Load observations for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: keyword arguments passed to `utils.load_data`
    """
    return load_data(path, sep=sep, **kwargs)
