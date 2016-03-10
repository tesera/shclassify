import os
import pytest
import pandas as pd

from shclassify import load_observations, load_model, DATA_DIR
from shclassify.utils import generate_fake_observations

def test_load_observations_raises_if_bad_path():
    with pytest.raises(OSError):
        load_observations('badpath')

def test_load_observations():
    df = load_observations('badpath')
    assert type(df) is pd.DataFrame

@pytest.mark.parametrize('model_filename', ['model_v-nv-wt.txt',
                                            'model_hf-hg-sc-so.txt',
                                            'model_f-nf.txt'])
def test_load_model(model_filename):
    model_path = os.path.join(DATA_DIR, model_filename)
    df = load_model(model_path)
    assert type(df) is pd.DataFrame

def test_generate_data():
    fake = generate_fake_observations(shape=(2,3))
    assert type(fake) is pd.DataFrame
    print(fake)
