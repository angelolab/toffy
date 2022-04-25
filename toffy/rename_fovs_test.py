# import tempfile

# from ark.utils import test_utils
from toffy import rename_fovs


def test_check_unnamed_fovs():
    ex_name = ['fov-1', 'fov-2', 'fov-3', 'fov-4']
    ex_run_order = [1, 2, 3, 4, 5]
    ex_scan_count = [1, 2, 3, 4, 5]
    ex_fov_list = []
    sample_fov_scan = {"fovs": ex_fov_list}

    for run_order, scan_count in zip(ex_run_order, ex_scan_count):
        if not run_order > len(ex_name):
            ex_fov = {
                "scanCount": scan_count,
                "runOrder": run_order,
                "name": ex_name[run_order-1]
            }
        else:
            ex_fov = {
                "scanCount": scan_count,
                "runOrder": run_order,
            }
        ex_fov_list.append(ex_fov)

    print(sample_fov_scan)
    rename_fovs.check_unnamed_fovs(sample_fov_scan)
