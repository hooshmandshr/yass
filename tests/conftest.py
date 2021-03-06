import os

import pytest


@pytest.fixture(scope='session')
def path_to_tests():
    path = os.path.dirname(os.path.realpath(__file__))
    return path


@pytest.fixture(scope='session')
def path_to_data_folder():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, 'data/')
