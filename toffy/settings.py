# tiled regions
REGION_PARAM_FIELDS = ['region_start_row', 'region_start_col', 'fov_num_row', 'fov_num_col',
                       'row_fov_size', 'col_fov_size', 'region_rand']

# mibitracker
MIBITRACKER_BACKEND = 'https://backend-dot-mibitracker-angelolab.appspot.com'

# co-registration
FIDUCIAL_POSITIONS = ['top left', 'top right', 'middle left', 'middle right',
                      'bottom left', 'bottom right']
MICRON_TO_STAGE_X_MULTIPLIER = 0.001001
MICRON_TO_STAGE_X_OFFSET = 0.3116
MICRON_TO_STAGE_Y_MULTIPLIER = 0.001018
MICRON_TO_STAGE_Y_OFFSET = 0.6294

# default stage to optical co-registration conversion params
STAGE_TO_OPTICAL_X_MULTIPLIER = 1 / 0.06887
STAGE_TO_OPTICAL_X_OFFSET = 27.79
STAGE_TO_OPTICAL_Y_MULTIPLIER = 1 / -0.06926
STAGE_TO_OPTICAL_Y_OFFSET = -77.40
