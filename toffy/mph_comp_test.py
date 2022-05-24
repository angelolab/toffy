import pytest
import os

from toffy import mph_comp as mph

def get_estimated_time():
    bad_path = os.path.join("data", "not-a-folder")
    bad_fov = "not-a-fov"

    good_path = os.path.join("data", "tissue")
    good_fov = 'fov-1-scan-1'

    # bad run file data should raise an error
    with pytest.raises(ValueError):
        mph.get_estimated_time(bad_path, good_fov)

    # bad run file data should raise an error
    with pytest.raises(ValueError):
        mph.get_estimated_time(good_path, bad_fov)

    # bad run file data should raise an error


    # test sucessful time data retrieval
    assert mph.get_estimated_time(good_path, good_fov) == 512
