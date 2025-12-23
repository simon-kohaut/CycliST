"""This file contains the Validator class to check a scene for any collisions or margin violations."""

# Standard Library
from typing import Any
from copy import deepcopy

# Third Party
import numpy as np

# CycliST
from cyclist.utility import get_frame


class Validator:
    """The CycliST scene validator class.

    The Validator is initialized from a scene configuration and, by calling its indivudal
    methods, validates the scene, checking if any margins violations or collisions occur.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/cyclist.py
    """

    @staticmethod
    def pairwise_distances(objects: list[dict[str, Any]]) -> list[float]:
        distances = []
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i != j:
                    distances.append(np.sqrt(
                        (objects[i]["location"]["x"] - objects[j]["location"]["x"]) ** 2
                        + (objects[i]["location"]["y"] - objects[j]["location"]["y"]) ** 2
                    ))

        return distances

    @staticmethod
    def is_collision_free(
        scene_config: dict[str, Any], objects: list[dict[str, Any]] | None = None
    ) -> bool:
        if objects is None:
            objects = scene_config["objects"]

        for distance in Validator.pairwise_distances(objects):
            if distance < scene_config['minimum_distance']:
                return False

        return True

    @staticmethod
    def is_always_collision_free(
        scene_config: dict[str, Any], objects: dict[str, Any] | None = None
    ) -> bool:
        if objects is None:
            objects = scene_config["objects"]

        for frame in range(int(scene_config["fps"] * scene_config["duration"])):
            frame_objects = get_frame(frame, objects)
            if not Validator.is_collision_free(scene_config, frame_objects):
                return False

        return True
