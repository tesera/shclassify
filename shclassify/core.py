from .utils import calc_num_na, load_data


def load_model(path, sep=',', index_col=[1], **kwargs):
    """Load a model for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: additional keyword arguments passed to `utils.load_data`
    """
    model = load_data(path, sep=sep, index_col=index_col, **kwargs)

    missing = calc_num_na(model)
    if missing > 0:
        raise ValueError(('Model has %s missing coefficients.' %missing))

    return model


def load_observations(path, sep=',', **kwargs):
    """Load observations for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: additional keyword arguments passed to `utils.load_data`
    """
    return load_data(path, sep=sep, **kwargs)


def calculate_logit(observations, model):
    """Apply model to observations

    .. warning:: for the model to be applied correctly, its row indices must be
    present among the column indices of the observations. If the correspondence
    is not meaninful (e.g. integer indices matched by coincidence), the
    result will not be meaninful!

    :param observations: `pandas.DataFrame` of observations
    :param model: `pandas.DataFrame` of model

    """
    variables = model.index()
    #: subset observations columns to indices of

    #: sort observations columns by index

    #: sort model rows by index

    #: multiply

    #: return result
