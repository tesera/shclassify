import os
import pytest
import pandas as pd
import numpy as np

from shclassify import (load_observations, load_model,
                        DATA_DIR, generate_fake_observations, calculate_prob,
                        choose_class_from_probs)
from shclassify.utils import (inverse_logit,
                              choose_from_multinomial_probs,
                              choose_from_binary_probs)
from shclassify.core import MODEL_FILES

def test_load_observations_raises_if_bad_path():
    with pytest.raises(OSError):
        load_observations('badpath')

@pytest.mark.xfail(reason='data not available yet')
def test_load_observations():
    df = load_observations('badpath')
    assert type(df) is pd.DataFrame

@pytest.mark.parametrize('model_filename', MODEL_FILES)
def test_load_model(model_filename):
    model_path = os.path.join(DATA_DIR, model_filename)
    df = load_model(model_path)
    assert type(df) is pd.DataFrame

def test_generate_data():
    fake = generate_fake_observations(2)
    assert type(fake) is pd.DataFrame
    assert fake.shape[1] == len(set(fake.columns.values))
    assert '(Intercept)' not in list(fake.columns.values)

@pytest.mark.parametrize('model_filename', MODEL_FILES)
def test_calculate_prob(model_filename):
    obs = generate_fake_observations(1)
    model_path = os.path.join(DATA_DIR, model_filename)
    model = load_model(model_path)
    probs = calculate_prob(obs, model)

    assert type(probs) is pd.DataFrame
    # result has shape of N_OBS, N_CLASSES
    assert probs.shape == (obs.shape[0],model.shape[1])

def test_inverse_logit():
    assert inverse_logit(0) == 0.5

def test_choose_from_multinomial_probs():
    n_obs = 3
    classes = ['a', 'b', 'c']
    df = pd.DataFrame(
        np.random.uniform(size=(n_obs,len(classes))), columns=classes
    )

    classes = choose_from_multinomial_probs(df)

    assert type(classes) is pd.DataFrame
    assert classes.shape == (n_obs, 1)
    assert classes.columns == ['class']

    def in_classes(x, classes=classes):
        x in classes

    assert classes['class'].apply(in_classes).all()

def test_choose_from_multinomial_probs_with_bad_input():
    n_obs = 3
    classes = ['a']
    df = pd.DataFrame(
        np.random.uniform(size=(n_obs,len(classes))), columns=classes
    )

    with pytest.raises(ValueError) as e:
        choose_from_multinomial_probs(df)

    assert 'Data frame must have more than 1 column' in str(e.value)

def test_choose_from_binary_probs():
    n_obs = 3
    df = pd.DataFrame(
        np.random.uniform(size=(n_obs,1))
    )

    classes = choose_from_binary_probs(df, 'true', 'false')

    assert type(classes) is pd.DataFrame
    assert classes.shape == (n_obs, 1)
    assert classes.columns == ['class']

def test_choose_from_binary_probs_with_bad_shape():
    n_obs = 3
    classes = ['a', 'b']
    df = pd.DataFrame(
        np.random.uniform(size=(n_obs,len(classes))), columns=classes
    )

    with pytest.raises(ValueError) as e:
        choose_from_binary_probs(df, 'true', 'false')

    assert 'Data frame must have 1 column' == str(e.value)

def test_choose_from_binary_probs_with_bad_args():
    n_obs = 3
    df = pd.DataFrame(
        np.random.uniform(size=(n_obs,1))
    )

    with pytest.raises(ValueError) as e:
        classes = choose_from_binary_probs(df, 'true', 'true')

    assert 'Class names for true and false results must differ' == str(e.value)

    with pytest.raises(ValueError) as e:
        classes = choose_from_binary_probs(df, 'true', 'false', threshold=50)

    assert 'Threshold must be between 0 and 1' == str(e.value)

@pytest.mark.xfail(message='Thin wrapper around binary and multinomial choice')
def test_choose_calss_from_probs():
    assert False
