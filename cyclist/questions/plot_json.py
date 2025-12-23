import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# read json
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

# plot
def plot_orbiting_objects(data, elev, azim, title,zlabel=True):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    directions = data['directions']
    objects = data['objects']
    relationships_seq = data['relationships_seq']

    left = np.array(directions['left'])
    right = np.array(directions['right'])
    front = np.array(directions['front'])
    behind = np.array(directions['behind'])
    up = np.cross(front, left)  # Correct the up vector calculation
    rotation_matrix = np.array([front, right]).T

    # reference line
    for center_index, obj in enumerate(objects):
        if obj.get('orbit', 'False') == 'False' and obj.get('linear', 'False') == 'False':
            center_coords = obj['3d_coords'][:2]
            center_coords = rotate_vector(center_coords, rotation_matrix)
            center_x, center_y, center_z = center_coords
            ax.plot([center_x]*2, [center_y]*2, [0, len(relationships_seq)], color="black", linestyle="--")
            break

    # set color label
    colors = {
        'blue': 'Left Front',
        'red': 'Right Front',
        'green': 'Left Behind',
        'yellow': 'Right Behind',
        'purple':'Right',
        'cyan':'Left',
        'orange':'Behind',
        'brown':'front'
    }
    for color, position in colors.items():
        ax.plot([], [], color=color, label=position)


    for index, obj in enumerate(objects):
        obj_name = obj['material'] +' '+ obj['shape']
        orbit = obj.get('orbit', 'False')
        linear = obj.get('linear', 'False')
        if orbit != 'False' or linear != 'False':

            print(len(obj['coords_seq']))
            coords_seq = [np.array((a[0],a[1])) for a in obj['coords_seq']]
            coords_seq = [rotate_vector(coord, rotation_matrix) for coord in coords_seq]
            xs, ys, _ = zip(*coords_seq)
            zs = list(range(len(coords_seq)))
            print(zs)
            for frame_idx in range(1, len(relationships_seq)):
                frame_relationships = relationships_seq[frame_idx]
                color = 'gray'  # default color

                left = frame_relationships['left'][center_index]
                right = frame_relationships['right'][center_index]
                front = frame_relationships['front'][center_index]
                behind = frame_relationships['behind'][center_index]

                if index in left and index in front:
                    color = 'blue'
                elif index in right and index in front:
                    color = 'red'
                elif index in left and index in behind:
                    color = 'green'
                elif index in right and index in behind:
                    color = 'yellow'
                elif index in left:
                    color = 'cyan'
                elif index in right:
                    color = 'purple'
                elif index in front:
                    color = 'brown'
                elif index in behind:
                    color = 'orange'


                # connect points
                if abs(xs[frame_idx-1]- xs[frame_idx])< 1:
                    ax.plot([xs[frame_idx-1], xs[frame_idx]], [ys[frame_idx-1], ys[frame_idx]],
                            [zs[frame_idx-1], zs[frame_idx]], color=color)
                if not zlabel:
                    if frame_idx == len(relationships_seq) // 2:  # For example, label the midpoint line
                        ax.text(xs[frame_idx], ys[frame_idx], zs[frame_idx], obj_name, color='black')

    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Frame Number')
    ax.view_init(elev=elev, azim=azim)
    if not zlabel:
        ax.zaxis.set_visible(False)
        ax.set_zticks([])  # Remove the Z axis ticks
        ax.set_zlabel('')  # Remove the Z axis label
        ax.zaxis.set_visible(False)
    ax.dist = 8
    plt.title(title)
    ax.legend(title='Relative Position')
    plt.show()


#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_scene', default='../output/scenes/CycliST_new_000000.json', type=str, help='The path to the scene file to visualize.')

args = parser.parse_args()

#
data = load_json(args.input_scene)
plot_orbiting_objects(data,30,-90,'')
plot_orbiting_objects(data,90,-90,'',zlabel=False)
#Top-Down View (90 degrees) of Orbiting Objects
#add 2d plots
