import numpy as np
import pandas as pd

def create_rosetta_matrices(default_matrix, output_directory, multipliers, channels=None):
    """Creates a series of compensation matrices for evaluating coefficients
    Args:
        default_matrix (str): path to the rosetta matrix to use as the default
        multipliers (list): the range of values to multiply the default matrix by
            to get new coefficients
        channels (list | None): an optional list of channels to include in the multiplication. If
            only a subset of channels are specified, other channels will retain their values
            in all iterations. If None, all channels are included

    >>> create_rosetta_matrices('compensation_matrix_intensity_nosum.csv', r'/Users/cameron/PycharmProjects/RosettaBetty', [0.5, 1, 1.5], [117])
    None
    """

    # step 1: read in the default matrix
    comp_matrix = pd.read_csv(default_matrix, index_col=0) # pandas DataFrame
    row_labels = comp_matrix.index
    comp_channels = list(row_labels)
    matrix_rows = len(comp_matrix)
    matrix_columns = len(comp_matrix.iloc[0])
    column_features = list(comp_matrix.columns.values)

    # step 2: figure out which channels will be modified (rows)
    if channels is None:
        channels = comp_channels

    for i in channels:
        if i not in comp_channels:
            raise ValueError('Specified channel does not exist')

    # step 3: loop over each of the multipliers (anything the user inputs, make an output matrix)
    # step 4: modify the appropriate channels based on mulitiplier
    # step 5: save the modified matrix as original_name_multiplier
    for i in multipliers: # returns each comp_matrix value
        zero_matrix = np.zeros(shape=(matrix_rows+1, matrix_columns))
        modified_matrix = pd.DataFrame(zero_matrix[1:], index = row_labels, columns=column_features)
        for j in range(matrix_rows):
            for k in channels:
                if k in comp_channels:
                    channel_index = comp_channels.index(k)
                    modified_matrix.iloc[channel_index,0:] = comp_matrix.iloc[channel_index,0:] * i
            modified_matrix.iloc[j,0:] = comp_matrix.iloc[j,0:]
        df = pd.DataFrame(modified_matrix)
        df.to_csv(output_directory+'/Rosetta_Titration%s.csv' % (str(i)))


def test_create_rosetta_matrices():
    # step 1: create rosetta matrix
    random_matrix = np.random.randint(1, 100, size =[47,47])

    # step 2: save as csv
    df2 = pd.DataFrame(random_matrix)
    df2.to_csv(r'/Users/cameron/PycharmProjects/RosettaBetty/Random_matrix.csv')
    #test_matrix = pd.DataFrame(random_matrix[0:])


    # step 3: run create_rosetta_matrices using template matrix
    create_rosetta_matrices('Random_matrix.csv', r'/Users/cameron/PycharmProjects/RosettaBetty', [2])

    # step 4: check that output is correct
    test = pd.read_csv('Rosetta_Titration2.csv', index_col=0)  # pandas DataFrame
    out = test/random_matrix
    print(out)
