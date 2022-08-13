# tiled regions
REGION_PARAM_FIELDS = ['region_name', 'region_start_row', 'region_start_col',
                       'fov_num_row', 'fov_num_col', 'row_fov_size', 'col_fov_size', 'region_rand']

# mibitracker
MIBITRACKER_BACKEND = 'https://backend-dot-mibitracker-angelolab.appspot.com'

# co-registration
COREG_SAVE_PATH = 'C:\\Users\\Customer.ION\\Documents\\toffy\\toffy\\coreg_params.json'
FIDUCIAL_POSITIONS = ['top left', 'top right', 'middle left', 'middle right',
                      'bottom left', 'bottom right']

MICRON_TO_STAGE_X_MULTIPLIER = 0.001001
MICRON_TO_STAGE_X_OFFSET = 0.3116
MICRON_TO_STAGE_Y_MULTIPLIER = 0.001018
MICRON_TO_STAGE_Y_OFFSET = 0.6294

STAGE_LEFT_BOUNDARY = -1.03
STAGE_RIGHT_BOUNDARY = 23.19
STAGE_TOP_BOUNDARY = 56.16
STAGE_BOTTOM_BOUNDARY = 1.57
OPTICAL_LEFT_BOUNDARY = 386.0
OPTICAL_RIGHT_BOUNDARY = 730.1
OPTICAL_TOP_BOUNDARY = 302.1
OPTICAL_BOTTOM_BOUNDARY = 1085.7

COREG_PARAM_BASELINE = {
    'STAGE_TO_OPTICAL_X_MULTIPLIER': 15.05,
    'STAGE_TO_OPTICAL_X_OFFSET': 26.15,
    'STAGE_TO_OPTICAL_Y_MULTIPLIER': -15.03,
    'STAGE_TO_OPTICAL_Y_OFFSET': -76.16
}


# QC channels to ignore
QC_CHANNEL_IGNORE = ['Au', 'Fe', 'Na', 'Ta', 'Noodle']

# QC metric .csv suffix and column naming
QC_SUFFIXES = ['nonzero_mean_stats', 'total_intensity_stats', 'percentile_99_9_stats']
QC_COLUMNS = ['Non-zero mean intensity', 'Total intensity', '99.9% intensity value']
