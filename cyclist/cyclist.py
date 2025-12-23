"""The CycliST main file for controlling the dataset generation."""

# Standard Library
import subprocess
import json
import os

# Third Party
import numpy as np

# CycliST
from .scene.generator import Generator
import sys
from rtpt import RTPT
import time
from .scene.cycle import Cycle
from .utility import get_frame


class CycliST:
    """CycliST - A benchmark for video question answering on cyclical state transitions.

    Args:
        scene_config: The scene configuration setting properties of the videos, objects and cycles.
            See 'cyclist.utility.parse_scene_config' section for reference.
    """

    def __init__(self, scene_config: dict[str, any]) -> "CycliST":
        self.scene_config = scene_config
        self.generator = Generator(scene_config)
        self.cycle = Cycle(scene_config)

    def run(self) -> bool:
        # Set the scene's own random seed (base seed + scene_index)
        np.random.seed(self.scene_config["seed"] + self.scene_config["scene_index"])

        # Create randomized objects in scene
        while not self.generator.generate():
            if not self.scene_config['force_generation']:
                return False

            self.generator.restart()

        # Setup lights
        if self.scene_config['cyclic_lights']:
            self.cycle.apply_light()

        # Indicate that scene has not been rendered yet
        self.scene_config["rendered"] = False

        # Store scene configuration
        self.scene_config["scene_config_file"] = (
            f"{self.scene_config['split']}_{self.scene_config['scene_index']}_config.json"
        )
        self.scene_config["video_file"] = (
            f"{self.scene_config['split']}_{self.scene_config['scene_index']}.mp4"
        )
        self.scene_config["blend_file"] = (
            f"{self.scene_config['split']}_{self.scene_config['scene_index']}.blend"
        )

        # Write-out pre-rendering scene configuration
        scene_config_file_path = os.path.join(
            self.scene_config["scene_config_directory"],
            self.scene_config["scene_config_file"],
        )
        with open(scene_config_file_path, "w") as scene_config_file:
            json.dump(self.scene_config, scene_config_file, indent=2)

        # Render video
        print("Start rendering ...")
        completed = subprocess.run(
            [
                "blender",
                "--background",
                "--python",
                "cyclist/scene/renderer.py",
                "--",
                "--scene_config",
                self.scene_config['scene_config_directory'] + "/" + self.scene_config['scene_config_file'],
            ],
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            print(completed.stderr)

        # Update scene config with render info (directions, time, rendered indicator,...)
        with open(scene_config_file_path, "r") as scene_config_file:
            self.scene_config = json.load(scene_config_file)
        # print(f"... took {self.scene_config['render_time']} seconds")

        # Create spatial labels (behind, left, right, in front of) and write back to disk
        self.assign_spatial_relations()

        # Create region labels (True if always within scene boundaries, else False)
        self.assign_region_labels()

        # Write final scene config to disk
        with open(scene_config_file_path, "w") as scene_config_file:
            json.dump(self.scene_config, scene_config_file, indent=2)

        return True

    def assign_spatial_relations(self):
        """Assign spatial labels (left, right, behind, front of) for each object pair."""

        self.scene_config["relationships"] = {}
        for name, direction in self.scene_config["directions"].items():
            # We store for each relationship separate dictionaries
            self.scene_config["relationships"][name] = {}

            # We do not assign relationships for above and below in CycliST
            if name == "above" or name == "below":
                continue

            # Iterate over frames and get for each the relationships
            # Each is stored as pair with the first being, e.g., left of, the second
            for frame in range(
                int(self.scene_config["fps"] * self.scene_config["duration"])
            ):
                self.scene_config["relationships"][name][frame] = []
                objects = get_frame(frame, self.scene_config["objects"])

                for i in range(len(objects)):
                    for j in range(len(objects)):
                        if i == j:
                            continue

                        difference = np.array(
                            [
                                objects[j]["location"]["x"]
                                - objects[i]["location"]["x"],
                                objects[j]["location"]["y"]
                                - objects[i]["location"]["y"],
                            ]
                        )
                        difference /= np.linalg.norm(difference)

                        dot_product = (
                            difference[0] * direction[0] + difference[1] * direction[1]
                        )

                        if dot_product > self.scene_config["relationship_threshold"]:
                            self.scene_config["relationships"][name][frame].append(
                                (i, j)
                            )

    def assign_region_labels(self) -> None:
        """Assigns for each object if it stays within the scenes main region."""

        # Assume that each object is within the scenes boundaries per default
        for obj in self.scene_config["objects"]:
            obj["always_within_boundaries"] = True

        # Check for each frame if the boundaries have been crossed
        for frame in range(
            int(self.scene_config["fps"] * self.scene_config["duration"])
        ):
            objects = get_frame(frame, self.scene_config["objects"])

            for index, obj in enumerate(objects):
                if (
                    obj["location"]["x"] < self.scene_config["min_x"]
                    or obj["location"]["x"] > self.scene_config["max_x"]
                    or obj["location"]["y"] < self.scene_config["min_y"]
                    or obj["location"]["y"] > self.scene_config["max_y"]
                ):
                    self.scene_config["objects"][index][
                        "always_within_boundaries"
                    ] = False


if __name__ == "__main__":
    # Parse command line arguments
    from copy import deepcopy
    from .utility import parse_scene_config

    # Setup CycliST with CLI args
    scene_config = parse_scene_config()

    print("render ", scene_config['number_of_videos'], "videos")
    rtpt = RTPT(name_initials='DO', experiment_name='Cyclist render', max_iterations=scene_config['number_of_videos'])
    rtpt.start()
    import torch
    t = torch.tensor([0])
    t.to("cuda")

    # Render all the scenes
    for scene_index in range(scene_config["number_of_videos"]):
        # Each scene gets its own index
        scene_config["scene_index"] = scene_index + scene_config["scene_index_offset"]

        # Generate the scene, including its config, video and questions
        CycliST(scene_config=deepcopy(scene_config)).run()
