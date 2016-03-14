import os
import pytest

from shclassify import DATA_DIR, Tree, generate_fake_observations


def data_file_path(modelfile):
    return os.path.join(DATA_DIR, modelfile)

@pytest.fixture
def tree_args():
    args = ('', data_file_path('model_v-nv-wt.txt')), \
           ('V', data_file_path('model_f-nf.txt')), \
           ('NF', data_file_path('model_hf-hg-sc-so.txt'))
           #('NV', data_file_path('model_v-nv-wt.txt')), \

    return args

@pytest.fixture
def path_to_observations_file():
    return data_file_path('observations.txt')

# TODO: use a smaller file here. best to manually create this in test
@pytest.fixture
def path_to_fake_observations_file():
    return data_file_path('fake_obs_1M.txt')

@pytest.fixture
def tree(tree_args):
    return Tree(*tree_args)

@pytest.fixture(scope='session')
def fake_observations():
    return generate_fake_observations(10)

@pytest.fixture(scope='session')
def fake_observation(fake_observations):
    return fake_observations.ix[[0],:]
