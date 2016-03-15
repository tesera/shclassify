import os
import logging
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict

from .utils import (calc_num_na, load_data, inverse_logit,
                    choose_from_binary_probs, choose_from_multinomial_probs)
from .config import DATA_DIR, MODEL_FILES


log = logging.getLogger(__name__)
MODEL_PATHS = [os.path.join(DATA_DIR,item) for item in MODEL_FILES]


def data_file_path(modelfile):
    return os.path.join(DATA_DIR, modelfile)

_default_model = (
    ('', {
        'model': data_file_path('model_v-nv-wt.txt')
    }),
    ('V', {
        'model': data_file_path('model_f-nf.txt'),
        'label_above_threshold': 'NF',
        'label_below_threshold': 'F',
        'threshold': 0.5
    }),
    ('NF', {
        'model': data_file_path('model_hf-hg-sc-so.txt')
    })
)

def load_model(path, sep=',', index_col=[0], **kwargs):
    """Load a model for SLS HRIS LCC Prediction

    :param path: path to input data, string or URL (e.g. http, s3)
    :param sep: input data field separator
    :param kwargs: additional keyword arguments passed to `utils.load_data`

    """
    log.info('Loading model at %s' %path)
    model = load_data(path, sep=sep, index_col=index_col, **kwargs)

    log.info('Checking model for missing coefficients')
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
    log.info('Loading observations at %s' %path)
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


def generate_fake_observations_file(n, path):
    """Generate fake observatiosn file for SLS HRIS LCC Prediction

    :param n: number of observations
    :param path: path to file

    """
    df = generate_fake_observations(n)
    df.to_csv(path)


def calculate_prob(observations, model, intercept_name='(Intercept)'):
    """Apply model to observations

    .. warning:: for the model to be applied correctly, its row indices must be present among the column indices of the observations. If the correspondence is not meaninful (e.g. indices matched by coincidence), the result will not be meaningful or an exception will be raised!

    :param observations: `pandas.DataFrame` of observations
    :param model: `pandas.DataFrame` of model
    :param intercept_name: name of the intercept field in the model

    """
    model_variables = model.index.tolist()
    intercept_index = model_variables.index(intercept_name)
    model_variables.pop(intercept_index)

    #TODO: isolate block below
    n_obs = observations.shape[0]
    log.info('Subsetting observations to variables in model')
    observations_for_model = observations.loc[:,model_variables]

    model_vars_set = set(model_variables)
    obs_vars_set = set(observations.columns.tolist())
    if not model_vars_set.issubset(obs_vars_set):
        set_diff = model_vars_set.difference(obs_vars_set)
        raise ValueError(
            'Observations are missing variables: {}'.format(set_diff)
        )

    observations_for_model.loc[:,intercept_name] = pd.Series(
        np.ones(n_obs), index=observations_for_model.index
    )
    observations_for_model.set_index([intercept_name], append=True)

    #: multiply - note that pandas handles checking index names match.
    #: use np.dot(X,B) for non-matching index names
    log.debug('Caclulating logits')
    result = observations_for_model.dot(model)
    log.info('Calculating probabilities')
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

    Construct a classification tree from logistic regression models

    :param tuples: tuples which define classification tree - see below for example
    :param path: path to model configuration file
    :param kwargs: keyword arguments to shclassify.load_model; currently assumes all model files have same tabular formatting

    The default model is

    ::

        ('', {
            'model': data_file_path('model_v-nv-wt.txt')
        }),
        ('V', {
            'model': data_file_path('model_f-nf.txt'),
            'label_above_threshold': 'NF',
            'label_below_threshold': 'F',
            'threshold': 0.5
        }),
        ('NF', {
            'model': data_file_path('model_hf-hg-sc-so.txt')
        })

    """
    def __init__(self, *tuples, path=None, **kwargs):
        self.config_filepath = path

        if path is not None:
            tuples = self._config_file_to_tuples(path)

            if tuples:
                raise ValueError(('Only one of path or tuples '
                                  'should be provided'))

        if not tuples:
            tuples = _default_model

        self._init_from_tuples(*tuples, **kwargs)

    def _config_file_to_tuples(self, path):
        raise RuntimeError('Not yet implemented')

    def _init_from_tuples(self, *tuples, **kwargs):
        tree = OrderedDict()

        for label, model_dict in tuples:
            if label in tree.keys():
                raise ValueError(('class (%s) must not have more than'
                                  ' one model assignment'))
            # TODO: validate that true and false labels provided for
            #       binary models
            tree[label] = {
                'model': load_model(model_dict['model']),
                'label_above_threshold': model_dict.get(
                    'label_above_threshold', '%s True' %label),
                'label_below_threshold': model_dict.get(
                    'label_below_threshold', '%s False' %label),
                'threshold': model_dict.get('threshold')
            }

        self.model = tree

    def predict_df(self, df):
        """Make predictions for observations in data frame

        .. note:: predictions will have same index as `df`

        :param df: `pandas.DataFrame` of observations

        """
        preds = pd.DataFrame(data='', index=df.index, columns=['class'])

        for cls, model_dict in self.model.items():
            model = model_dict['model']
            # remove null to avoid conflict with next mask query
            mask = pd.notnull(preds['class'])
            if not mask.any():
                log.debug('Stopping prediction for class %s as all values are null' %cls)
                break

            # subset to parent class of model
            mask = mask & (preds['class']==cls)
            if not mask.any():
                log.debug('Stopping prediction because no observatiosn are class %s' %cls)
                break

            obs = df[mask.values]
            obs_cls_probs = calculate_prob(obs, model)
            cls_pred = choose_class_from_probs(
                obs_cls_probs,
                name_true=model_dict['label_above_threshold'],
                name_false=model_dict['label_below_threshold'],
                threshold=model_dict['threshold']
            )
            preds[mask] = cls_pred

        return preds

    def predict_file(self, obs_file, pred_file, overwrite=False,
                     sep=',', chunksize=10000, index_col=None):
        """Make predictions for observations in file

        This automatically appends predictions to `pred_file`. To get
        predictions in a data frame in interactive python sessions, use
        `predict_df`.

        :param obs_file: path to observations file
        :param pred_file: path of file to write predictions
        :param overwrite: overwrite `pred_file` if it exists
        :param sep: observation file separator
        :param chunksize: chunksize (lines) read `pred_file` for making predictions
        :param index_col: integer index of column to use as data frame row index
        :param kwargs: keyword arguments to `load_observations`
        """
        reader = load_observations(obs_file, sep=sep,
                                   chunksize=chunksize,
                                   index_col=index_col)

        if os.path.exists(pred_file) and not overwrite:
            raise ValueError('%s already exists! Specify a new file.')

        for i, chunk in enumerate(reader):
            log.info('Loading chunk %s' %i)
            res = self.predict_df(chunk)
            mode = 'w' if i==0 else 'a'
            header = mode == 'w'

            res.to_csv(pred_file, header=header, mode=mode)
