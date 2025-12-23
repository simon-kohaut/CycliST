"""This file contains the CycliST scene generator, a class to create randomized CycliST scenes."""

# Standard Library
from typing import Any
from copy import deepcopy
import os

# Third Party
import numpy as np

# CycliST
from .cycle import Cycle
from .validator import Validator


class Generator:
    """The CycliST scene generator class.

    The Generator is initialized from a scene configuration and, by calling its indivudal
    methods, extends the scene with new objects or cycles.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/cyclist.py
    """

    def __init__(self, scene_config: dict[str, Any]) -> "Generator":
        self.cycle = Cycle(scene_config)
        self.validator = Validator()
        self.scene_config = scene_config

    @staticmethod
    def get_random_asset(directory: str) -> str:
        """Selects a random asset, e.g., mesh or material, from a directory.

        Args:
            directory: The directory to select an asset from

        Returns:
            The name of a random asset from the chosen directory including its file extension
        """

        assets = []
        for file in os.listdir(directory):
            if file.endswith(".blend"):
                assets.append(os.path.splitext(file)[0])

        return str(np.random.choice(assets))

    def restart(self) -> None:
        """Clears all generated info from the scene."""

        del self.scene_config['objects']

    def generate(self) -> bool:
        """Generates a scene according to the provided configuration.

        This method updates the scene_config of the Generator object.

        Returns:
            False if the scene could not be generated, e.g., without violating margins, else True
        """

        # Autocomplete what has been provided beforehand
        # So the user can decide some guarantees
        predetermined = deepcopy(self.scene_config["objects"])
        self.scene_config["objects"] = []
        for object_config in predetermined:
            print(f"Autocomplete pre-determined object with {object_config} ...")
            success = False
            for number_of_tries in range(self.scene_config["max_number_of_tries"]):
                trial = deepcopy(object_config)
                if self.generate_object(trial):
                    print(f"... took {number_of_tries + 1} tries")
                    self.add_object(trial)
                    success = True
                    break

        # Add the static clutter objects
        if self.scene_config["number_of_clutter_objects"] is None:
            self.scene_config["number_of_clutter_objects"] = np.random.randint(
                self.scene_config["min_number_of_clutter_objects"],
                self.scene_config["max_number_of_clutter_objects"] + 1,
            )

        print(
            f"Generate {self.scene_config["number_of_clutter_objects"]} clutter objects."
        )
        for _ in range(self.scene_config["number_of_clutter_objects"]):
            print(f"Generate clutter object ...")

            success = False
            for number_of_tries in range(self.scene_config["max_number_of_tries"]):
                object_config = {}
                if self.generate_object(object_config):
                    print(f"... took {number_of_tries + 1} tries")
                    self.add_object(object_config)
                    success = True
                    break

            if not success:
                print(f"... failed to insert within {number_of_tries + 1} tries!")
                return False

        # Decide how many cycles of each type will be generated
        if self.scene_config["number_of_resize_cycles"] is None:
            self.scene_config["number_of_resize_cycles"] = np.random.randint(
                self.scene_config["min_number_of_resize_cycles"],
                self.scene_config["max_number_of_resize_cycles"] + 1,
            )

        if self.scene_config["number_of_orbit_cycles"] is None:
            self.scene_config["number_of_orbit_cycles"] = np.random.randint(
                self.scene_config["min_number_of_orbit_cycles"],
                self.scene_config["max_number_of_orbit_cycles"] + 1,
            )

        if self.scene_config["number_of_recolor_cycles"] is None:
            self.scene_config["number_of_recolor_cycles"] = np.random.randint(
                self.scene_config["min_number_of_recolor_cycles"],
                self.scene_config["max_number_of_recolor_cycles"] + 1,
            )

        if self.scene_config["number_of_linear_cycles"] is None:
            self.scene_config["number_of_linear_cycles"] = np.random.randint(
                self.scene_config["min_number_of_linear_cycles"],
                self.scene_config["max_number_of_linear_cycles"] + 1,
            )

        if self.scene_config["number_of_rotate_cycles"] is None:
            self.scene_config["number_of_rotate_cycles"] = np.random.randint(
                self.scene_config["min_number_of_rotate_cycles"],
                self.scene_config["max_number_of_rotate_cycles"] + 1,
            )

        self.scene_config["max_number_of_cyclic_objects"] = (
            self.scene_config["number_of_resize_cycles"]
            + self.scene_config["number_of_orbit_cycles"]
            + self.scene_config["number_of_recolor_cycles"]
            + self.scene_config["number_of_linear_cycles"]
            + self.scene_config["number_of_rotate_cycles"]
        )

        # Decide which objects will do which cycles
        orbit_indices = set(
            np.random.choice(
                range(self.scene_config["max_number_of_cyclic_objects"]),
                size=self.scene_config["number_of_orbit_cycles"],
                replace=False,
            )
        )
        linear_indices = set(
            np.random.choice(
                list(
                    set(range(self.scene_config["max_number_of_cyclic_objects"]))
                    - orbit_indices
                ),
                size=self.scene_config["number_of_linear_cycles"],
                replace=False,
            )
        )
        resize_indices = set(
            np.random.choice(
                range(self.scene_config["max_number_of_cyclic_objects"]),
                size=self.scene_config["number_of_resize_cycles"],
                replace=False,
            )
        )
        recolor_indices = set(
            np.random.choice(
                range(self.scene_config["max_number_of_cyclic_objects"]),
                size=self.scene_config["number_of_recolor_cycles"],
                replace=False,
            )
        )
        rotate_indices = set(
            np.random.choice(
                range(self.scene_config["max_number_of_cyclic_objects"]),
                size=self.scene_config["number_of_rotate_cycles"],
                replace=False,
            )
        )

        # For each cyclic object, we generate with the randomly assigned cycles
        object_configs = []
        for index in range(self.scene_config["max_number_of_cyclic_objects"]):
            object_config = {}
            cycles = {}
            if index in orbit_indices:
                cycles["orbit"] = {}
            if index in linear_indices:
                cycles["linear"] = {}
            if index in resize_indices:
                cycles["resize"] = {}

                # TODO: Remove this part
                # Ensure that resizing objects start large
                # object_config["size"] = "large"
            if index in recolor_indices:
                cycles["recolor"] = {}
            if index in rotate_indices:
                cycles["rotate"] = {}

                # Ensure that rotate objects are not spheres
                object_config["mesh"] = "Sphere"
                while object_config["mesh"] == "Sphere":
                    object_config["mesh"] = self.get_random_asset(
                        self.scene_config["mesh_directory"]
                    )

            # If no cycles where assigned, drop the object
            if cycles == {}:
                continue

            object_config["cycles"] = cycles
            object_configs.append(object_config)

        for object_config in object_configs:
            # Ignore orbiting objects the first time around since they need possible center objects to exist first
            if "orbit" in object_config["cycles"].keys():
                continue

            print(f"Generate cyclic object with {object_config} ...")
            success = False
            for number_of_tries in range(self.scene_config["max_number_of_tries"]):
                trial = deepcopy(object_config)
                if self.generate_object(trial):
                    print(f"... took {number_of_tries + 1} tries")
                    self.add_object(trial)
                    success = True
                    break

            if not success:
                print(f"... failed to insert within {number_of_tries + 1} tries!")
                return False

        for object_config in object_configs:
            # Ignore non-orbiting objects since we already added them above
            if not "orbit" in object_config["cycles"].keys():
                continue

            print(f"Generate cyclic object with {object_config} ...")
            success = False
            for number_of_tries in range(self.scene_config["max_number_of_tries"]):
                trial = deepcopy(object_config)
                if self.generate_object(trial):
                    print(f"... took {number_of_tries + 1} tries")
                    self.add_object(trial)
                    success = True
                    break

            if not success:
                print(f"... failed to insert within {number_of_tries + 1} tries!")
                return False

        return True

    def add_object(self, object_config: dict[str, Any]) -> None:
        """Add an object to the scene.

        Args:
            object_config: A dictionary describing the objects properties
        """

        # Ensure there is space in the config to append the object to and insert
        if "objects" not in self.scene_config.keys():
            self.scene_config["objects"] = []
        self.scene_config["objects"].append(object_config)

    def generate_object(self, object_config: dict[str, Any]) -> bool:
        """Generate an object for the scene.

        The method follows the idea that everything is generated randomly if
        the respective key is not in the provided configuration. For example,
        if the location key is not in the object_config, a random location is
        generated, letting the caller decide what parts should be deterministic
        and what is random.

        Args:
            object_config: A dictionary describing the objects properties

        Returns:
            False if the object could not be generated, e.g., without violating margins, else True
        """

        # Decide random mesh if none was given
        if "mesh" not in object_config.keys():
            object_config["mesh"] = self.get_random_asset(
                self.scene_config["mesh_directory"]
            )

        # Decide random material if none was given
        if "material" not in object_config.keys():
            object_config["material"] = self.get_random_asset(
                self.scene_config["material_directory"]
            )

        # Decide random size if none was given
        if "size" not in object_config.keys():
            object_config["size"] = str(
                np.random.choice(list(self.scene_config["sizes"].keys()))
            )

        # Decide random color if none was given
        if "color" not in object_config.keys():
            object_config["color"] = str(
                np.random.choice(list(self.scene_config["colors"].keys()))
            )

        # Add all the cycles, returning False if any single one was unable to be added
        if "cycles" in object_config.keys():
            self.cycle.apply(object_config)

        # Decide random orientation if none was given
        if "angle" not in object_config.keys():
            object_config["angle"] = 2 * np.pi * np.random.uniform(0, 1)

        # Decide random location if none was given
        if "location" not in object_config.keys():
            # Set a random location
            object_config["location"] = dict(
                {
                    "x": float(
                        np.random.uniform(
                            self.scene_config["min_x"], self.scene_config["max_x"]
                        )
                    ),
                    "y": float(
                        np.random.uniform(
                            self.scene_config["min_y"], self.scene_config["max_y"]
                        )
                    ),
                    "z": self.scene_config["sizes"][object_config["size"]],
                }
            )

        # Check if scene is valid if the object is placed at that location
        if "objects" in self.scene_config.keys():
            objects = self.scene_config["objects"] + [object_config]
        else:
            objects = [object_config]

        return self.validator.is_always_collision_free(self.scene_config, objects)
