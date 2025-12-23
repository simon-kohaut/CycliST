"""Utility functions not part of any scene generation class."""

# Standard Library
from copy import deepcopy
from typing import Any
from argparse import ArgumentParser
import json

# Third Party
import numpy as np


def get_frame(frame: int, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns a dictionary of all the objects at a certain frame."""

    frame_objects = []
    for obj in objects:
        if "cycles" in obj.keys() and "linear" in obj["cycles"].keys():
            frame_objects.append(
                {
                    "location": obj["cycles"]["linear"]["states"][frame]["location"],
                    "size": obj["size"],
                }
            )
        elif "cycles" in obj.keys() and "orbit" in obj["cycles"].keys():
            frame_objects.append(
                {
                    "location": obj["cycles"]["orbit"]["states"][frame]["location"],
                    "size": obj["size"],
                }
            )
        else:
            frame_objects.append(
                {
                    "location": obj["location"],
                    "size": obj["size"],
                }
            )

    return frame_objects


def parse_scene_config(args: list[str] | None = None) -> dict[str, Any]:
    """Gets a scene config from the provided arguments or the command line.

    Args:
        args: The vector of arguments to parse; uses sys.args if None was provided
    """

    parser = ArgumentParser()

    # Basic information, usually the same for all scenes, e.g., directory structure
    parser.add_argument(
        "--base_scene_path",
        default="assets/scene.blend",
        type=str,
        help="The base scene.blend to render in.",
    )
    parser.add_argument(
        "--material_directory",
        default="assets/materials",
        type=str,
        help="The directory containing materials.",
    )
    parser.add_argument(
        "--mesh_directory",
        default="assets/shapes",
        type=str,
        help="The directory containing meshes.",
    )
    parser.add_argument(
        "--colors_path",
        default="assets/colors.json",
        type=str,
        help="The path to the possible color names and values.",
    )
    parser.add_argument(
        "--sizes_path",
        default="assets/sizes.json",
        type=str,
        help="The path to the possible sizes names and values.",
    )
    parser.add_argument(
        "--write_blendfile",
        action='store_true',
        help="Whether the scene's blendfile should be written to disk.",
    )
    parser.add_argument(
        "--blendfile_path",
        default="output/blendfiles",
        type=str,
        help="Where to write the scene's blendfile to.",
    )
    parser.add_argument(
        "--max_number_of_tries",
        default=100,
        type=int,
        help="How many tries to insert an object.",
    )
    parser.add_argument(
        "--force_generation",
        action='store_true',
        help="Whether to force generating a video even if max number of tries was reached."
        "Restarts the scene generation with the same parameters and scene index until a valid scene is obtained.",
    )

    # Video and scene properties
    parser.add_argument(
        "--scene_config_directory",
        default="output/scenes",
        type=str,
        help="The path to the scene configuration.",
    )
    parser.add_argument(
        "--video_directory",
        default="output/videos",
        type=str,
        help="The path to the rendered video.",
    )
    parser.add_argument(
        "--scene_index",
        default=0,
        type=int,
        help="An offset on the scene's index.",
    )
    parser.add_argument(
        "--scene_index_offset",
        default=0,
        type=int,
        help="An offset on the scene's index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The randomization seed which will be combined with the scene index."
        "So if the seed is 1 and scene index 2, the random state will be set to 3."
        "If None is given, a random seed is chosen.",
    )
    parser.add_argument(
        "--fps", default=32, type=int, help="The video's frames per second."
    )
    parser.add_argument(
        "--duration", default=5.0, type=float, help="The video's duration in seconds."
    )
    parser.add_argument(
        "--number_of_videos",
        default=1,
        type=int,
        help="The number of videos to render.",
    )
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="The name of the dataset split being rendered.",
    )
    parser.add_argument(
        "--min_x",
        default=-5.0,
        type=float,
        help="The minimum coordinate on the x-axis.",
    )
    parser.add_argument(
        "--max_x", default=5.0, type=float, help="The maximum coordinate on the x-axis."
    )
    parser.add_argument(
        "--min_y",
        default=-5.0,
        type=float,
        help="The minimum coordinate on the y-axis.",
    )
    parser.add_argument(
        "--max_y", default=5.0, type=float, help="The maximum coordinate on the y-axis."
    )
    parser.add_argument(
        "--relationship_threshold",
        default=0.5,
        type=float,
        help="The threshold to decide if a spatial relationship holds based on the dot product between object displacement and cardinal relationship direction.",
    )
    parser.add_argument(
        "--minimum_distance",
        default=1.5,
        type=float,
        help="The minimum distance between objects to avoid collisions and spatial ambiguity.",
    )

    # Rendering information
    parser.add_argument(
        "--device",
        default="CUDA",
        type=str,
        help="Which device the video is rendered on.",
    )
    parser.add_argument(
        "--resolution_width",
        default=1920,
        type=int,
        help="The horizontal resolution of the video.",
    )
    parser.add_argument(
        "--resolution_height",
        default=1080,
        type=int,
        help="The vertical resolution of the video.",
    )
    parser.add_argument(
        "--number_of_samples",
        default=4,
        type=int,
        help="How many samples blender takes for rendering.",
    )
    parser.add_argument(
        "--min_number_of_bounces",
        default=8,
        type=int,
        help="How many times light bounces at least.",
    )
    parser.add_argument(
        "--max_number_of_bounces",
        default=8,
        type=int,
        help="How many times light bounces at most.",
    )

    # Lights and Camera
    parser.add_argument(
        "--cyclic_lights",
        action='store_true',
        help="Whether the lights will cyclically change intensity.",
    )
    parser.add_argument(
        "--key_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the key light position.",
    )
    parser.add_argument(
        "--fill_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the fill light position.",
    )
    parser.add_argument(
        "--back_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the back light position.",
    )
    parser.add_argument(
        "--camera_jitter",
        default=0.5,
        type=float,
        help="The magnitude of random jitter to add to the camera position",
    )

    # Clutter objects
    parser.add_argument(
        "--min_number_of_clutter_objects",
        default=0,
        type=int,
        help="The minimum number of clutter objects.",
    )
    parser.add_argument(
        "--max_number_of_clutter_objects",
        default=0,
        type=int,
        help="The maximum number of clutter objects.",
    )
    parser.add_argument(
        "--number_of_clutter_objects",
        type=int,
        help="The number of clutter objects.",
    )

    # Cyclic objects
    parser.add_argument(
        "--min_number_of_orbit_cycles",
        default=0,
        type=int,
        help="The minimum number of orbiting objects.",
    )
    parser.add_argument(
        "--max_number_of_orbit_cycles",
        default=0,
        type=int,
        help="The maximum number of orbiting objects.",
    )
    parser.add_argument(
        "--number_of_orbit_cycles",
        type=int,
        help="The number of orbiting objects.",
    )
    parser.add_argument(
        "--min_number_of_rotate_cycles",
        default=0,
        type=int,
        help="The minimum number of rotating objects.",
    )
    parser.add_argument(
        "--max_number_of_rotate_cycles",
        default=0,
        type=int,
        help="The maximum number of rotating objects.",
    )
    parser.add_argument(
        "--number_of_rotate_cycles",
        type=int,
        help="The number of rotating objects.",
    )
    parser.add_argument(
        "--min_number_of_resize_cycles",
        default=0,
        type=int,
        help="The minimum number of resizing objects.",
    )
    parser.add_argument(
        "--max_number_of_resize_cycles",
        default=0,
        type=int,
        help="The maximum number of resizing objects.",
    )
    parser.add_argument(
        "--number_of_resize_cycles",
        type=int,
        help="The number of resizing objects.",
    )
    parser.add_argument(
        "--min_number_of_linear_cycles",
        default=0,
        type=int,
        help="The minimum number of linear moving objects.",
    )
    parser.add_argument(
        "--max_number_of_linear_cycles",
        default=0,
        type=int,
        help="The maximum number of linear moving objects.",
    )
    parser.add_argument(
        "--number_of_linear_cycles",
        type=int,
        help="The number of linear objects.",
    )
    parser.add_argument(
        "--min_number_of_recolor_cycles",
        default=0,
        type=int,
        help="The minimum number of recoloring objects.",
    )
    parser.add_argument(
        "--max_number_of_recolor_cycles",
        default=0,
        type=int,
        help="The maximum number of recoloring objects.",
    )
    parser.add_argument(
        "--number_of_recolor_cycles",
        type=int,
        help="The number of recoloring objects.",
    )

    # Cycle properties
    parser.add_argument(
        "--min_number_of_prime_factors",
        default=5,
        type=int,
        help="The minimum number of prime factors of fps * duration factored into cycle duration.",
    )
    parser.add_argument(
        "--max_number_of_prime_factors",
        default=6,
        type=int,
        help="The maximum number of prime factors of fps * duration factored into cycle duration.",
    )

    scene_config = vars(parser.parse_args(args))

    # Append colors
    with open(scene_config["colors_path"], "r") as colors_file:
        scene_config["colors"] = json.load(colors_file)

    for color in scene_config["colors"].keys():
        scene_config["colors"][color] = [
            float(value) / 255.0 for value in scene_config["colors"][color]
        ] + [1.0]

    # Append sizes
    with open(scene_config["sizes_path"], "r") as sizes_file:
        scene_config["sizes"] = json.load(sizes_file)

    # If no seed was given, choose a random one
    if scene_config["seed"] is None:
        scene_config["seed"] = int(np.random.rand() * (2**32 - 1))

    return scene_config
