import os
import logging
import datetime
import numpy as np
import pandas as pd

from .utils import (calc_num_na, load_data, inverse_logit,
                    choose_from_binary_probs, choose_from_multinomial_probs)
from .config import DATA_DIR, MODEL_FILES


log = logging.getLogger(__name__)
MODEL_PATHS = [os.path.join(DATA_DIR,item) for item in MODEL_FILES]


def load_model(path, sep=',', index_col=[0], **kwargs):
    """Load a model for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: additional keyword arguments passed to `utils.load_data`

    """
    log.debug('Loading model at %s' %path)
    model = load_data(path, sep=sep, index_col=index_col, **kwargs)

    log.debug('Checking model for missing coefficients')
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
    log.debug('Loading observations at %s' %path)
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
    is not meaninful (e.g. indices matched by coincidence), the result will not
    be meaningful!

    :param observations: `pandas.DataFrame` of observations
    :param model: `pandas.DataFrame` of model
    :param intercept_name: name of the intercept field in the model

    """
    model_variables = model.index.tolist()
    intercept_index = model_variables.index(intercept_name)
    model_variables.pop(intercept_index)

    #TODO: isolate block below
    n_obs = observations.shape[0]
    log.debug('Subsetting observations to variables in model')
    observations_for_model = observations.ix[:,model_variables]
    observations_for_model.loc[:,intercept_name] = pd.Series(
        np.ones(n_obs), index=observations_for_model.index)
    observations_for_model.set_index([intercept_name], append=True)

    #: multiply - note that pandas handles checking index names match.
    #: use np.dot(X,B) for non-matching index names
    log.debug('Caclulating logits')
    result = observations_for_model.dot(model)
    log.debug('Calculating probabilities')
    result = result.applymap(inverse_logit)

    return result


def choose_class_from_probs(df, **kwargs):
    """Choose class from data frame of probabilities

    :param df: `pandas.DataFrame` of binary or multinomial class probabilities
    :param kwargs: keyword arguments to `utils.choose_from_binary_probs`

    """
    if df.shape[1] == 1:
        class_preds = choose_from_binary_probs(df, **kwargs)
    else:
        class_preds = choose_from_multinomial_probs(df)
    return class_preds


class Tree:
    """Model tree

    .. note:: kwargs assumes all modelfiles have same format

    .. note:: model file columns must map to class names

    :param *tuples: tuples of (class, modelfile) to
    :param path: path to model configuration file
    :param kwargs: keyword arguments to shclassify.load_model

    """
    def __init__(self, *tuples, path=None, **kwargs):
        self.config_filepath = path

        if path is not None:
            tuples = self._config_file_to_tuples(path)

            if tuples:
                raise ValueError(('Only one of path or tuples '
                                  'should be provided'))

        self._init_from_tuples(*tuples, **kwargs)
        self.depth = len(self.model.keys()) # TODO: one level can have 4
                                            #       models, this is wrong

    def _config_file_to_tuples(self, path):
        raise RuntimeError('Not yet implemented')

    def _init_from_tuples(self, *tuples, **kwargs):
        tree = {}

        for k, v in tuples:
            if k in tree.keys():
                raise ValueError('class %s has more than one model assignment')
            tree[k] = load_model(v)

        self.model = tree

    def _validate_level(self, level):
        if type(level) != int:
            raise TypeError('Level must be integer')
        if level > self.depth:
            raise ValueError(
                'Level (%s) cannot exceed tree depth (%s)' \
                %(level, self.depth)
            )

    def _predict_one(self, x, level=None):
        """ Make class predictions for one observation using tree model

        :param x: observation, row from `pandas.DataFrame`
        :param level: depth in tree to which to predict

        """
        x_class = ['']
        current_cls = x_class[-1]
        depth = self.depth
        # TODO: not ideal. predict_df uses this method for df.apply
        #       which operates on rows as series
        #       that is not compatible with calculate_prob, which is
        #       designed for use with data frames
        row = pd.DataFrame().append(x)

        if level is not None:
            self._validate_level(level)
            depth = level

        while current_cls in self.model and len(x_class) < depth:
            model = self.model[current_cls]
            probs_df = calculate_prob(row, model)
            cls_df = choose_class_from_probs(probs_df)
            # TODO: use something which returns str instead of DataFrame
            current_cls = cls_df.iloc[0,0]
            x_class.append(current_cls)

        return ' '.join(str(x_class))

    def predict_df(self, df):
        """Make predictions for observatiosn in data frame

        :param df: `pandas.DataFrame` of observations

        """
        level = None
        before = datetime.datetime.now()
        print(before)

        preds = df.apply(self._predict_one, axis=1, level=level)

        after = datetime.datetime.now()
        print(after)
        print(after-before)
        return pd.DataFrame(preds)

    def predict_file(self, obs_file, pred_file, overwrite=False,
                     sep=',', chunksize=1000, **kwargs):
        """Make predictions for observations in file

        This automatically appends predictions to `pred_file`. To get
        predictions in a data frame in interactive python sessions, use
        `predict_df`.

        :param obs_file: path to observations file
        :param pred_file: path of file to write predictions
        :param overwrite: overwrite `pred_file` if it exists
        :param sep: observation file separator
        :param chunksize: chunksize read `pred_file` for making predictions (MB)

        """
        reader = load_observations(obs_file, sep=sep,
                                   chunksize=chunksize, **kwargs)

        if os.path.exists(pred_file) and not overwrite:
            raise ValueError('%s already exists! Specify a new file.')

        for i, chunk in enumerate(reader):
            res = self.predict_df(chunk)
            mode = 'w' if i==0 else 'a'
            header = mode == 'w'
            print('************* %s ***************' %i)
            print(mode)

            res.to_csv(pred_file, header=header, mode=mode)
