import pytest
import pandas as pd

from shclassify import load_input_data


def test_load_input_data_raises_if_bad_path():
    with pytest.raises(OSError):
        load_input_data('badpath')

def test_load_input_data():
    df = load_input_data('badpath')
    assert type(df) is pd.DataFrame
