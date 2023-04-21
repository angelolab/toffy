import os
import shutil
import tempfile

import pytest
from alpineer import io_utils, load_utils, test_utils

from tests.utils.test_utils import make_run_file
from toffy import image_stitching


def test_get_max_img_size():
    # test with run file
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = make_run_file(tmp_dir, prefixes=[""])

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size("extracted_dir", run_dir=test_dir)
        assert max_img_size == 20

        # test success for fov list
        max_img_size = image_stitching.get_max_img_size(
            "extracted_dir", run_dir=test_dir, fov_list=["fov-2-scan-1", "fov-3-scan-1"]
        )
        assert max_img_size == 10

    # test by reading image sizes
    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ["Au", "CD3", "CD4", "CD8", "CD11c"]
        fov_list = ["fov-1-scan-1", "fov-2-scan-1"]
        larger_fov = ["fov-3-scan-1"]

        test_utils._write_tifs(tmpdir, fov_list, channel_list, (16, 16), "", False, int)
        test_utils._write_tifs(tmpdir, larger_fov, channel_list, (32, 32), "", False, int)

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size(tmpdir)
        assert max_img_size == 32

        # test success for fov list
        max_img_size = image_stitching.get_max_img_size(
            tmpdir, run_dir="bin_dir", fov_list=["fov-1-scan-1", "fov-2-scan-1"]
        )
        assert max_img_size == 16

    # test by reading image sizes in subdir
    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ["Au", "CD3", "CD4", "CD8", "CD11c"]
        fov_list = ["fov-1-scan-1", "fov-2-scan-1"]
        larger_fov = ["fov-3-scan-1"]

        test_utils._write_tifs(tmpdir, fov_list, channel_list, (16, 16), "sub_dir", False, int)
        test_utils._write_tifs(tmpdir, larger_fov, channel_list, (32, 32), "sub_dir", False, int)

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size(tmpdir, img_sub_folder="sub_dir")
        assert max_img_size == 32


@pytest.mark.parametrize("nontiled_fov", [False, True])
@pytest.mark.parametrize("prefixes", [[""], ["Tile_1_"], ["Tile_1_", "Tile_2_"]])
def test_get_tiled_names(prefixes, nontiled_fov):
    # test with run file
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = make_run_file(tmp_dir, prefixes=prefixes, include_nontiled=nontiled_fov)

        if len(prefixes) > 1:
            fov4_prefix = prefixes[1]
        else:
            fov4_prefix = prefixes[0]
        tiled_names = [f"{prefixes[0]}R1C3", f"{prefixes[0]}R2C1", f"{fov4_prefix}R2C2"]
        fov_list = ["fov-1-scan-1", "fov-2-scan-1", "fov-4-scan-1"]

        fov_names = image_stitching.get_tiled_names(fov_list, test_dir)
        assert list(fov_names.values()) == fov_list
        assert list(fov_names.keys()) == tiled_names

        # check for subset of fovs in run that actually have image dirs
        tiled_names = [f"{prefixes[0]}R1C3", f"{fov4_prefix}R2C2"]
        fov_list = ["fov-1-scan-1", "fov-4-scan-1"]

        fov_names = image_stitching.get_tiled_names(fov_list, test_dir)
        assert list(fov_names.values()) == fov_list
        assert list(fov_names.keys()) == tiled_names


@pytest.mark.parametrize(
    "tiled, tile_names",
    [
        (False, [""]),
        (True, [""]),
        (True, ["Tile_1_"]),
        (True, ["Tile_1_", "Tile_2_"]),
        (True, ["", "Tile_1_"]),
    ],
)
@pytest.mark.parametrize("nontiled_fov", [False, True])
@pytest.mark.parametrize("subdir", ["", "sub_name"])
def test_stitch_images(mocker, tiled, tile_names, nontiled_fov, subdir):
    mocker.patch("toffy.image_stitching.get_max_img_size", return_value=10)

    channel_list = ["Au", "CD3", "CD4", "CD8", "CD11c"]
    stitched_tifs = [
        "Au_stitched.tiff",
        "CD3_stitched.tiff",
        "CD4_stitched.tiff",
        "CD8_stitched.tiff",
        "CD11c_stitched.tiff",
    ]
    fov_num = 4
    if nontiled_fov:
        fov_num = 5
    fov_list = [f"fov-{i}-scan-1" for i in range(1, fov_num + 1)]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_name = os.path.basename(tmpdir)
        if tiled:
            stitched_dir = f"{run_name}_tiled"
        else:
            stitched_dir = f"{run_name}_stitched"

        test_dir = make_run_file(tmpdir, prefixes=tile_names, include_nontiled=nontiled_fov)
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), subdir, False, int)

        # bad channel should raise an error
        with pytest.raises(
            ValueError,
            match="Not all values given in list channel inputs were found in list valid channels.",
        ):
            image_stitching.stitch_images(
                tmpdir, test_dir, ["Au", "bad_channel"], img_sub_folder=subdir, tiled=tiled
            )

        # test successful stitching for all channels
        image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
        if tiled:
            if "" in tile_names:
                tile_names = ["unnamed_tile_" if x == "" else x for x in tile_names]
            # multiple tiles
            if len(tile_names) > 1:
                # check subfolder for each tile
                for i, tile in enumerate(tile_names):
                    save_dir = os.path.join(tmpdir, stitched_dir, tile[:-1])
                    assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
                    data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
                    if i == 0:
                        assert data.shape == (1, 20, 30, 1)
                    else:
                        assert data.shape == (1, 20, 20, 1)
            # single tile
            else:
                save_dir = os.path.join(tmpdir, stitched_dir, tile_names[0][:-1])
                assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
                data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
                assert data.shape == (1, 20, 30, 1)

        # max img size 10 with 4 or 5 acquired fovs
        else:
            save_dir = os.path.join(tmpdir, stitched_dir)
            assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
            data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
            if nontiled_fov:
                assert data.shape == (1, 30, 20, 1)
            else:
                assert data.shape == (1, 20, 20, 1)

        # test previous stitching raises an error
        with pytest.raises(ValueError, match="The stitch_images subdirectory already exists"):
            image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
        shutil.rmtree(os.path.join(tmpdir, stitched_dir))

        # test stitching for specific channels
        image_stitching.stitch_images(
            tmpdir, test_dir, channels=["Au", "CD3"], img_sub_folder=subdir, tiled=tiled
        )
        if tiled and len(tile_names) > 1:
            # check each tile subfolder
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, tile_names[0][:-1]))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, tile_names[1][:-1]))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
        elif tiled:
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, tile_names[0][:-1]))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
        else:
            assert sorted(io_utils.list_files(os.path.join(tmpdir, stitched_dir))) == sorted(
                ["Au_stitched.tiff", "CD3_stitched.tiff"]
            )
        shutil.rmtree(os.path.join(tmpdir, stitched_dir))

    with tempfile.TemporaryDirectory() as tmpdir:
        run_name = os.path.basename(tmpdir)
        stitched_dir = f"{run_name}_stitched"
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), subdir, False, int)

        # test stitching for no run_dir
        if not tiled:
            image_stitching.stitch_images(
                tmpdir, channels=["Au", "CD3"], img_sub_folder=subdir, tiled=tiled
            )
            assert sorted(io_utils.list_files(os.path.join(tmpdir, stitched_dir))) == sorted(
                ["Au_stitched.tiff", "CD3_stitched.tiff"]
            )
        else:
            with pytest.raises(ValueError):
                image_stitching.stitch_images(
                    tmpdir,
                    channels=["Au", "CD3"],
                    img_sub_folder=subdir,
                    tiled=tiled,
                )
