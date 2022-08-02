# tiled regions
REGION_PARAM_FIELDS = ['region_name', 'region_start_row', 'region_start_col',
                       'fov_num_row', 'fov_num_col', 'row_fov_size', 'col_fov_size', 'region_rand']

# mibitracker
MIBITRACKER_BACKEND = 'https://backend-dot-mibitracker-angelolab.appspot.com'

# co-registration
FIDUCIAL_POSITIONS = ['top left', 'top right', 'middle left', 'middle right',
                      'bottom left', 'bottom right']
MICRON_TO_STAGE_X_MULTIPLIER = 0.001001
MICRON_TO_STAGE_X_OFFSET = 0.3116
MICRON_TO_STAGE_Y_MULTIPLIER = 0.001018
MICRON_TO_STAGE_Y_OFFSET = 0.6294

# QC channels to ignore
QC_CHANNEL_IGNORE = ['Au', 'Fe', 'Na', 'Ta', 'Noodle']

# QC metric .csv suffix and column naming
QC_SUFFIXES = ['nonzero_mean_stats', 'total_intensity_stats', 'percentile_99_9_stats']
QC_COLUMNS = ['Non-zero mean intensity', 'Total intensity', '99.9% intensity value']
