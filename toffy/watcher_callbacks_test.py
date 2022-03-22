from toffy import watcher_callbacks


def test_build_extraction_and_qc_callback():
    global_panel = (-0.3, 0.0)

    # test cb generates w/o errors
    _ = watcher_callbacks.build_extract_and_compute_qc_callback(global_panel)

    # TODO: add small bin files to test cb
    pass
