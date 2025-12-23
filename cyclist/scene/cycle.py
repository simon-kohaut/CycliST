"""Implements the cyclical state transitions."""

# Standard Library
from typing import Any

# Third Party
import numpy as np


class Cycle:

    def __init__(self, scene_config: dict[str, Any]) -> "Cycle":
        self.scene_config = scene_config

    @staticmethod
    def prime_factors(n: int) -> list[int]:
        """Computes the prime factors of a number.

        Args:
            n: The number to compute the prime factors for

        Returns:
            A list of the prime factors of 'n'
        """

        primes = []

        while n % 2 == 0:
            primes.append(2)
            n //= 2

        for i in range(3, int(np.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes.append(i)
                n //= i

        if n > 2:
            primes.append(n)

        return primes

    def choose_period(self) -> int:
        """Chooses a cycle period from random  prime factors of the total number of frames.

        Returns:
            The cycle period in frames (number of frames until object is back in original state)
        """

        primes = self.prime_factors(
            self.scene_config["fps"] * self.scene_config["duration"]
        )
        factors = np.random.choice(
            primes,
            size=np.random.randint(
                self.scene_config["min_number_of_prime_factors"],
                self.scene_config["max_number_of_prime_factors"] + 1,
            ),
            replace=False,
        )

        return int(np.prod(factors))

    def apply_orbit(self, object_config: dict[str, Any]):
        assert (
            "objects" in self.scene_config.keys()
            and len(self.scene_config["objects"]) > 0
        ), "Orbiting objects require already existing objects in the scene."

        # Get cycle period in frames
        period = self.choose_period()

        # Initial angle and its increment (randomly chosen clockwise or counter clockwise)
        initial_angle = np.random.uniform(0, 2.0 * np.pi)
        angle_increment = np.random.choice([-1.0, 1.0]) * 2.0 * np.pi / period

        # Store information about orbit direction
        object_config["orbit_direction"] = "clockwise" if angle_increment < 0.0 else "counterclockwise"

        # Select random object as center of orbit
        object_config["center"] = int(
            np.random.choice(range(len(self.scene_config["objects"])))
        )
        center = self.scene_config["objects"][object_config["center"]]
        radius = float(
            np.random.uniform(self.scene_config["sizes"][center["size"]] * 2.0, 5.0)
        )

        # Utility function for location dependent on the frame
        def orbiting_location(center_x: float, center_y: float, frame: int):
            return {
                "x": float(
                    center_x + radius * np.cos(initial_angle + angle_increment * frame)
                ),
                "y": float(
                    center_y + radius * np.sin(initial_angle + angle_increment * frame)
                ),
            }

        # Set the object's locations through time
        states = []
        for base_frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):
            for cycle_frame in range(period):
                frame = base_frame + cycle_frame

                if "cycles" in center.keys() and "orbit" in center["cycles"].keys():
                    center_x = float(
                        center["cycles"]["orbit"]["states"][frame - 1]["location"]["x"]
                    )
                    center_y = float(
                        center["cycles"]["orbit"]["states"][frame - 1]["location"]["y"]
                    )
                elif "cycles" in center.keys() and "linear" in center["cycles"].keys():
                    center_x = float(
                        center["cycles"]["linear"]["states"][frame - 1]["location"]["x"]
                    )
                    center_y = float(
                        center["cycles"]["linear"]["states"][frame - 1]["location"]["y"]
                    )
                else:
                    center_x = float(center["location"]["x"])
                    center_y = float(center["location"]["y"])

                states.append(
                    {
                        "location": orbiting_location(center_x, center_y, cycle_frame),
                        "frame": frame,
                    }
                )

        object_config["cycles"]["orbit"]["period"] = period
        object_config["cycles"]["orbit"]["states"] = states

    def apply_linear(self, object_config: dict[str, Any]):
        # Get cycle period in frames
        period = self.choose_period()

        # Generate start and intermittent location
        object_config["location"] = {
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
        object_config["intermittent_location"] = {
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

        # Get the distance the object will move in a single frame
        distance = np.array(
            [
                object_config["intermittent_location"]["x"]
                - object_config["location"]["x"],
                object_config["intermittent_location"]["y"]
                - object_config["location"]["y"],
            ]
        )

        # Set the object's locations through time
        states = []
        for base_frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):
            # Moving towards intermittent location
            for cycle_frame in range(period):
                if cycle_frame <= period / 2.0:
                    displacement = cycle_frame / (period / 2.0) * distance
                # Arrived there and moving back again
                else:
                    displacement = (
                        distance
                        - (cycle_frame - period / 2.0) / (period / 2.0) * distance
                    )

                # The location for the frame
                states.append(
                    {
                        "location": {
                            "x": object_config["location"]["x"] + displacement[0],
                            "y": object_config["location"]["y"] + displacement[1],
                        },
                        "frame": base_frame + cycle_frame,
                    }
                )

        object_config["cycles"]["linear"]["period"] = period
        object_config["cycles"]["linear"]["states"] = states

    def apply_resize(self, object_config: dict[str, Any]):
        # Get cycle period in frames and select other size
        period = self.choose_period()
        object_config["intermittent_size"] = str(
            np.random.choice(
                [
                    color
                    for color in self.scene_config["sizes"].keys()
                    if color != object_config["size"]
                ]
            )
        )

        # Set size for beginning, middle and end of a cycle and let Blender interpolate
        states = [{"factor": 1.0, "frame": 1}]
        size_ratios = (
            self.scene_config["sizes"][object_config["intermittent_size"]]
            / self.scene_config["sizes"][object_config["size"]],
            self.scene_config["sizes"][object_config["size"]]
            / self.scene_config["sizes"][object_config["intermittent_size"]],
        )
        for frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):
            states.append(
                {
                    "factor": size_ratios[0],
                    "frame": frame + period // 2,
                }
            )
            states.append(
                {
                    "factor": size_ratios[1],
                    "frame": frame + period,
                }
            )

        object_config["cycles"]["resize"]["period"] = period
        object_config["cycles"]["resize"]["states"] = states

    def apply_recolor(self, object_config: dict[str, Any]):
        # Get cycle period in frames and select intermittent color
        period = self.choose_period()
        object_config["intermittent_color"] = str(
            np.random.choice(
                [
                    color
                    for color in self.scene_config["colors"].keys()
                    if color != object_config["color"]
                ]
            )
        )

        # Set colors for beginning, middle and end of a cycle and let Blender interpolate
        states = []
        for frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):

            states.append(
                {
                    "color": object_config["color"],
                    "frame": frame,
                }
            )
            states.append(
                {
                    "color": object_config["intermittent_color"],
                    "frame": frame + period // 2,
                }
            )
            states.append(
                {
                    "color": object_config["color"],
                    "frame": frame + period,
                }
            )

        object_config["cycles"]["recolor"]["period"] = period
        object_config["cycles"]["recolor"]["states"] = states

    def apply_rotate(self, object_config: dict[str, Any]):
        # Ger period and angle difference per frame
        period = self.choose_period()

        angle_per_frame = np.zeros(3)
        angle_per_frame[0] = 2.0 * np.pi / period

        # Set angle difference for each frame for smooth transitions
        states = []
        for base_frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):
            for cycle_frame in range(period):
                states.append(
                    {
                        "rotation": list(cycle_frame * angle_per_frame),
                        "frame": base_frame + cycle_frame,
                    }
                )

        object_config["cycles"]["rotate"]["period"] = period
        object_config["cycles"]["rotate"]["states"] = states

    def apply_light(self) -> None:
        """Make the lights go through a day-night cycle."""
        
        # Ger lighting period
        period = self.choose_period()
        
        # Set light intensity for each frame
        states = []
        for base_frame in range(
            1, int(self.scene_config["fps"] * self.scene_config["duration"]) + 1, period
        ):
            for cycle_frame in range(period):
                states.append(
                    {
                        "intensity": 0.5 * np.cos((cycle_frame / period) * (2 * np.pi)) + 0.5,
                        "frame": base_frame + cycle_frame,
                    }
                )

        self.scene_config['lights'] = {
            'period': period,
            'states': states,
        }

    def apply(self, object_config: dict[str, Any]) -> None:
        if "recolor" in object_config["cycles"]:
            self.apply_recolor(object_config)
        if "rotate" in object_config["cycles"]:
            self.apply_rotate(object_config)
        if "resize" in object_config["cycles"]:
            self.apply_resize(object_config)
        if "linear" in object_config["cycles"]:
            self.apply_linear(object_config)
        if "orbit" in object_config["cycles"]:
            self.apply_orbit(object_config)
