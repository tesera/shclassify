import os
import numpy as np
import pandas as pd

from .utils import calc_num_na, load_data, inverse_logit
from .config import DATA_DIR, MODEL_FILES


MODEL_PATHS = [os.path.join(DATA_DIR,item) for item in MODEL_FILES]


def load_model(path, sep=',', index_col=[0], **kwargs):
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


def get_features_from_model_files(paths, features_col=0,
                                  intercept_names=['(Intercept)']):
    """Retrieve features from model files

    :param paths: paths to model files
    :param kwargs: keyword arguments to `load_model`
    :param intercept_names: list of strings to be removed from features
    """
    features = []
    for path in paths:
        model = load_model(path, usecols=[features_col],
                           index_col=None)
        features += model.values.flatten().tolist()

    features = set(features)
    features = features.difference(set(intercept_names))

    return features


# TODO: compatibility with custom models
def generate_fake_observations(n):
    """Generate fake data for SLS HRIS LCC Prediction

    :param n: number of observations to generate
    """
    features = get_features_from_model_files(MODEL_PATHS)

    n_col = len(features)
    arr = np.random.uniform(size=(n, n_col))
    df = pd.DataFrame(data=arr)
    df.columns = features
    return df


# TODO: may want to copy observations as it gets mutated
#     : or ensure that there are no ill side effects
def calculate_prob(observations, model, intercept_name='(Intercept)'):
    """Apply model to observations

    .. warning:: for the model to be applied correctly, its row indices must be
    present among the column indices of the observations. If the correspondence
    is not meaninful (e.g. integer indices matched by coincidence), the
    result will not be meaninful!

    :param observations: `pandas.DataFrame` of observations
    :param model: `pandas.DataFrame` of model
    :param intercept_name: name of the intercept field in the model

    """
    model_variables = model.index.tolist()
    intercept_index = model_variables.index(intercept_name)
    model_variables.pop(intercept_index)

    #TODO: isolate block below
    #: df[['names']] *copies*
    #: pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    n_obs = observations.shape[0]
    observations_for_model = observations[model_variables]
    observations_for_model[intercept_name] = np.ones(n_obs)
    observations_for_model.set_index([intercept_name], append=True)

    #: multiply - note that pandas handles checking index names match.
    #: use np.dot(X,B) for non-matching index names
    result = observations_for_model.dot(model)
    result = result.applymap(inverse_logit)

    return result
