import os
import shutil
import tempfile

import pytest
from alpineer import io_utils, load_utils, test_utils

from toffy import image_stitching, json_utils


def make_run_file(tmp_dir, multiple_tiles=False):
    """Create a run subir and run json in the provided dir and return the path to this new dir."""

    if multiple_tiles:
        prefix1, prefix2 = ["Tile_1_", "Tile_2_"]
    else:
        prefix1, prefix2 = ["Tile_1_"] * 2

    RUN_JSON_SPOOF = {
        "fovs": [
            {
                "runOrder": 1,
                "scanCount": 1,
                "frameSizePixels": {"width": 32, "height": 32},
                "name": f"{prefix1}R1C3",
            },
            {
                "runOrder": 2,
                "scanCount": 1,
                "frameSizePixels": {"width": 16, "height": 16},
                "name": f"{prefix1}R2C1",
            },
            {
                "runOrder": 3,
                "scanCount": 1,
                "frameSizePixels": {"width": 8, "height": 8},
                "name": f"{prefix1}MoQC",
            },
            {
                "runOrder": 4,
                "scanCount": 1,
                "frameSizePixels": {"width": 16, "height": 16},
                "name": f"{prefix2}R2C2",
            },
            {
                "runOrder": 5,
                "scanCount": 1,
                "frameSizePixels": {"width": 16, "height": 16},
                "name": f"non-tiled",
            },
        ],
    }

    test_dir = os.path.join(tmp_dir, "data", "test_run")
    os.makedirs(test_dir)
    json_path = os.path.join(test_dir, "test_run.json")
    json_utils.write_json_file(json_path, RUN_JSON_SPOOF)

    return test_dir


def test_get_max_img_size():
    # test with run file
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = make_run_file(tmp_dir)

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size("extracted_dir", run_dir=test_dir)
        assert max_img_size == 32

        # test success for fov list
        max_img_size = image_stitching.get_max_img_size(
            "extracted_dir", run_dir=test_dir, fov_list=["fov-2-scan-1", "fov-3-scan-1"]
        )
        assert max_img_size == 16

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


@pytest.mark.parametrize("multiple", [False, True])
def test_get_tiled_names(multiple):
    # test with run file
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = make_run_file(tmp_dir, multiple_tiles=multiple)

        if multiple:
            prefix2 = "Tile_2_"
        else:
            prefix2 = "Tile_1_"
        fov_list = ["fov-1-scan-1", "fov-2-scan-1", "fov-4-scan-1"]
        tiled_names = ["Tile_1_R1C3", "Tile_1_R2C1", f"{prefix2}R2C2"]

        fov_names = image_stitching.get_tiled_names(fov_list, test_dir)
        assert list(fov_names.values()) == fov_list
        assert list(fov_names.keys()) == tiled_names

        # check for subset of fovs in run that actually have image dirs
        fov_list = ["fov-1-scan-1", "fov-4-scan-1"]
        tiled_names = ["Tile_1_R1C3", f"{prefix2}R2C2"]

        fov_names = image_stitching.get_tiled_names(fov_list, test_dir)
        assert list(fov_names.values()) == fov_list
        assert list(fov_names.keys()) == tiled_names


@pytest.mark.parametrize("tiled, multiple", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("subdir", ["", "sub_name"])
def test_stitch_images(mocker, tiled, multiple, subdir):
    mocker.patch("toffy.image_stitching.get_max_img_size", return_value=32)

    channel_list = ["Au", "CD3", "CD4", "CD8", "CD11c"]
    stitched_tifs = [
        "Au_stitched.tiff",
        "CD3_stitched.tiff",
        "CD4_stitched.tiff",
        "CD8_stitched.tiff",
        "CD11c_stitched.tiff",
    ]
    fov_list = ["fov-1-scan-1", "fov-2-scan-1", "fov-3-scan-1", "fov-4-scan-1", "fov-5-scan-1"]

    if tiled:
        stitched_dir = "stitched_images_tiled"
    else:
        stitched_dir = "stitched_images"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = make_run_file(tmpdir, multiple_tiles=multiple)
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
        if tiled:
            if multiple:
                image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
                for tile in ["Tile_1", "Tile_2"]:
                    save_dir = os.path.join(tmpdir, stitched_dir, tile)
                    assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
                    data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
                    if tile == "Tile_1":
                        assert data.shape == (1, 20, 30, 1)
                    else:
                        assert data.shape == (1, 20, 20, 1)
            # single tile
            else:
                save_dir = os.path.join(tmpdir, stitched_dir, "Tile_1")
                image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
                assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
                data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
                assert data.shape == (1, 20, 30, 1)

        # max img size 32 with 5 acquired fovs
        else:
            save_dir = os.path.join(tmpdir, stitched_dir)
            image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
            assert sorted(io_utils.list_files(save_dir)) == sorted(stitched_tifs)
            data = load_utils.load_imgs_from_dir(save_dir, files=["Au_stitched.tiff"])
            assert data.shape == (1, 96, 64, 1)

        # test previous stitching raises an error
        with pytest.raises(ValueError, match="The stitch_images subdirectory already exists"):
            image_stitching.stitch_images(tmpdir, test_dir, img_sub_folder=subdir, tiled=tiled)
        shutil.rmtree(os.path.join(tmpdir, stitched_dir))

        # test stitching for specific channels
        image_stitching.stitch_images(
            tmpdir, test_dir, channels=["Au", "CD3"], img_sub_folder=subdir, tiled=tiled
        )
        if tiled and multiple:
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, "Tile_1"))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, "Tile_2"))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
        elif tiled:
            assert sorted(
                io_utils.list_files(os.path.join(tmpdir, stitched_dir, "Tile_1"))
            ) == sorted(["Au_stitched.tiff", "CD3_stitched.tiff"])
        else:
            assert sorted(io_utils.list_files(os.path.join(tmpdir, stitched_dir))) == sorted(
                ["Au_stitched.tiff", "CD3_stitched.tiff"]
            )
        shutil.rmtree(os.path.join(tmpdir, stitched_dir))

    with tempfile.TemporaryDirectory() as tmpdir:
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
