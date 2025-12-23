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
        "CUDA",
        "--seed",
        "0",
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
        "0",
    ]
)

scene_config["objects"] = [
    {
        "location": {"x": 0, "y": 0},
        "mesh": "Cylinder",
        "material": "Rubber",
        "color": "red",
        "size": "large",
        "angle": 45.0,
        "cycles": {"rotate": {}}
    }
]

cyclist = CycliST(scene_config)
cyclist.run()
