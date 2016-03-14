import os
import pytest
import pandas as pd

from shclassify import Tree

def test_init_tree_from_tuples(tree_args):
    tree = Tree(*tree_args)

    assert tree.config_filepath is None
    for item in tree.model.values():
        assert type(item) == pd.DataFrame
    assert tree.depth == 4

@pytest.mark.xfail(message='not yet implemented')
def test_init_tree_with_invalid_model():
    assert False

@pytest.mark.xfail(message='not yet implemented')
def test_init_tree_from_file():
    assert False

@pytest.mark.xfail(message='not yet implemented')
def test_init_tree_from_invalid_file():
    assert False

@pytest.mark.xfail(message='not yet implemented')
def test_init_tree_with_duplicate_classes():
    assert False

def test__validate_level(tree):
    with pytest.raises(ValueError) as e:
        tree._validate_level(tree.depth+1)
        assert 'cannot exceed tree depth' in str(e.value)

    with pytest.raises(TypeError) as e:
        tree._validate_level('a')
        assert str(e.value) == 'Level must be integer'

def test__predict_one(tree, fake_observation):
    pred = tree._predict_one(fake_observation, level=None)
    assert type(pred) is str

def test_predict_df(tree, fake_observations):
    preds = tree.predict_df(fake_observations)
    assert type(preds) is pd.DataFrame
    print(preds)

def test_predict_file(tree, path_to_observations_file):
    outfile = os.path.join(os.getcwd(), 'predictions.txt')
    tree.predict_file(path_to_observations_file, outfile)
