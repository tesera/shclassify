import os
import pytest
import pandas as pd

from shclassify import (load_observations, load_model,
                        DATA_DIR, generate_fake_observations, calculate_prob,
                        choose_class_from_probs)
from shclassify.core import MODEL_FILES

def test_load_observations_raises_if_bad_path():
    with pytest.raises(OSError):
        load_observations('badpath')

def test_load_observations(path_to_observations_file):
    df = load_observations(path_to_observations_file)
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
    obs = generate_fake_observations(1000)
    model_path = os.path.join(DATA_DIR, model_filename)
    model = load_model(model_path)
    probs = calculate_prob(obs, model)

    assert type(probs) is pd.DataFrame
    # result has shape of N_OBS, N_CLASSES
    assert probs.shape == (obs.shape[0],model.shape[1])

@pytest.mark.xfail(message='Thin wrapper around binary and multinomial choice')
def test_choose_calss_from_probs():
    assert False
