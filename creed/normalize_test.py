import numpy as np
import pandas as pd

from creed import normalize


def test_create_objective_function():
    degree = 2

    func_name = 'poly_{}'.format(degree)

    obj_func = normalize.create_objective_function(func_name)

    inputs = [1] * (degree + 2)

    output = obj_func(*inputs)

    assert output == degree + 1


def test_combine_run_metrics():

    bins = []
    metrics = []
    for i in range(1, 5):
        bin_name = 'example_{}.bin'.format(i)
        metric_name = 'example_metric_{}.csv'.format(i)
        metric_values = {'column_1': np.random.rand(10),
                         'column_2': np.random.rand(10),
                         'column_3': np.random.rand(10)}


