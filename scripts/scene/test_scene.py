from cyclist.cyclist import CycliST
from cyclist.utility import parse_scene_config
from cyclist.scene.validator import Validator
import json

scene_config = parse_scene_config(
    [
        "--duration",
        "1",
        "--fps",
        "32",
        "--device",
        "METAL",
        "--write_blendfile",
        "--seed",
        "0",
        "--cyclic_lights",
        "--min_number_of_prime_factors",
        "5",
        "--max_number_of_prime_factors",
        "5",
        "--max_x",
        "5",
        "--min_x",
        "-5",
        "--max_y",
        "5",
        "--min_y",
        "-5",
        "--number_of_clutter_objects",
        "2",
        "--number_of_orbit_cycles",
        "2",
        "--number_of_linear_cycles",
        "2",
        "--number_of_resize_cycles",
        "2",
        "--number_of_recolor_cycles",
        "2",
    ]
)

scene_config["objects"] = [
    {
        "location": {"x": scene_config['min_x'], "y": scene_config['min_y']},
        "mesh": "Cube",
        "material": "Rubber",
        "color": "red",
        "size": "large",
        "angle": 0.0,
    },
    {
        "location": {"x": scene_config['max_x'], "y": scene_config['min_y']},
        "mesh": "Cube",
        "material": "Rubber",
        "color": "blue",
        "size": "large",
        "angle": 0.0,
    },
    {
        "location": {"x": scene_config['max_x'], "y": scene_config['max_y']},
        "mesh": "Cube",
        "material": "Rubber",
        "color": "green",
        "size": "large",
        "angle": 0.0,
    },
    {
        "location": {"x": scene_config['min_x'], "y": scene_config['max_y']},
        "mesh": "Cube",
        "material": "Rubber",
        "color": "yellow",
        "size": "large",
        "angle": 0.0,
    },
]

cyclist = CycliST(scene_config)
cyclist.run()

print("Initial spatial relations for objects:")
for i in range(len(cyclist.scene_config['objects'])):
    for j in range(len(cyclist.scene_config['objects'])):
        if i == j:
            continue

        for relation in cyclist.scene_config['relationships'].keys():
            if relation == 'above' or relation == 'below':
                continue

            for (target, source) in cyclist.scene_config['relationships'][relation][0]:
                if (target, source) == (i, j):
                    print(
                        f"{cyclist.scene_config['objects'][j]['color']} {cyclist.scene_config['objects'][j]['mesh']} "
                        f"is {relation} of {cyclist.scene_config['objects'][i]['color']} {cyclist.scene_config['objects'][i]['mesh']}."
                    )
