#

with open('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/code/ark-analysis/data/example_dataset/json_tiling_data/20210116_TMA3_corners.json', 'r') as flf:
    fov_list_info = json.load(flf)

new_fovs = []
top_left = copy.deepcopy(fov_list_info['fovs'][0])
top_left['centerPointMicrons']['x'] = 6979
top_left['centerPointMicrons']['y'] = 46554
top_left['name'] = 'top_left'
new_fovs.append(top_left)

top_right = copy.deepcopy(fov_list_info['fovs'][0])
top_right['centerPointMicrons']['x'] = 18448
top_right['centerPointMicrons']['y'] = 46554
top_right['name'] = 'top_right'

new_fovs.append(top_right)

bottom_left = copy.deepcopy(fov_list_info['fovs'][0])
bottom_left['centerPointMicrons']['x'] = 4979
bottom_left['centerPointMicrons']['y'] = 21448
bottom_left['name'] = 'bottom_left'

new_fovs.append(bottom_left)


bottom_right = copy.deepcopy(fov_list_info['fovs'][0])
bottom_right['centerPointMicrons']['x'] = 17448
bottom_right['centerPointMicrons']['y'] = 19448
bottom_right['name'] = 'bottom_right'

new_fovs.append(bottom_right)



new_json = {'fovs': new_fovs}
with open('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/code/ark-analysis/data/example_dataset/json_tiling_data/20210116_TMA3_4_corners_renamed2.json', 'w') as rtp:
    json.dump(new_json, rtp)

tma_generate_fov_list('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/code/ark-analysis/data/example_dataset/json_tiling_data/20210116_TMA3_4_corners.json',
                      6, 11)



@dataclass
class xy_coord:
    x: float
    y: float

top_left = xy_coord(new_json['fovs'][0]['centerPointMicrons']['x'],
                    new_json['fovs'][0]['centerPointMicrons']['y'])

top_right = xy_coord(new_json['fovs'][1]['centerPointMicrons']['x'],
                     new_json['fovs'][1]['centerPointMicrons']['y'])

bottom_left = xy_coord(new_json['fovs'][2]['centerPointMicrons']['x'],
                       new_json['fovs'][2]['centerPointMicrons']['y'])

bottom_right = xy_coord(new_json['fovs'][3]['centerPointMicrons']['x'],
                        new_json['fovs'][3]['centerPointMicrons']['y'])

x_y_pairs = generate_x_y_fov_pairs_rhombus(top_left, top_right, bottom_left, bottom_right,
                                           6, 11)

num_y = 11
num_x = 6

# compute shift in y across the top and bottom of the TMA
    top_y_shift = top_right.y - top_left.y
    bottom_y_shift = bottom_right.y - bottom_left.y

    # average between the two will be used to increment indices
    avg_y_shift = (top_y_shift + bottom_y_shift) / 2

    # compute shift in x across the sides of the tma
    left_x_shift = bottom_left.x - top_left.x
    right_x_shift = bottom_right.x - top_right.x

    # average between the two will be used to increment indices
    avg_x_shift = (left_x_shift + right_x_shift) / 2

    # compute per-FOV adjustment
    x_increment = avg_x_shift / (num_y - 1)
    y_increment = avg_y_shift / (num_x - 1)

    # compute baseline indices for a rectangle with same coords
    x_dif = bottom_right.x - top_left.x
    y_dif = bottom_right.y - top_left.y

    x_baseline = x_dif / (num_x - 1)
    y_baseline = y_dif / (num_y - 1)

    pairs = []

    for i in range(num_x):
        for j in range(num_y):
            x_coord = top_left.x + x_baseline * i + x_increment * j
            y_coord = top_left.y + y_baseline * j + y_increment * i
            pairs.append((int(x_coord), int(y_coord)))
