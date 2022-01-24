from creed import normalize


def test_create_objective_function():
    degree = 2

    func_name = 'poly_{}'.format(degree)

    obj_func = normalize.create_objective_function(func_name)

    inputs = [1] * (degree + 2)

    output = obj_func(*inputs)

    assert output == degree + 1

