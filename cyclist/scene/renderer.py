"""Supplies the rendering capabilities of CycliST usiing Blender.

This file is supposed to be called as main file through Blender, using bpy to define
the rendering pipeline. This will be triggered by running cyclist, but could me manually handled
by running blender --background --python cyclist/scene/renderer.py -- --scene_config /path/to/scene/config.json.
"""

# Standard Library
from typing import Any
import os
import json
from time import time

# Third Party
import numpy as np


def setup_rendering(scene_config: dict[str, Any]) -> None:
    """Sets all Blender parameters used for rendering.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    # Load base scene and materials
    bpy.ops.wm.open_mainfile(filepath=scene_config["base_scene_path"])
    for file in os.listdir(scene_config["material_directory"]):
        if file.endswith(".blend"):
            material_name = os.path.splitext(file)[0]
            material_path = os.path.join(
                scene_config["material_directory"], file, "NodeTree", material_name
            )
            bpy.ops.wm.append(filename=material_path)

    # Set engine and viewport parameters
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.filepath = (
        scene_config["video_directory"] + "/" + scene_config["video_file"]
    )
    bpy.context.scene.render.resolution_x = scene_config["resolution_width"]
    bpy.context.scene.render.resolution_y = scene_config["resolution_height"]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(scene_config["fps"] * scene_config["duration"])
    bpy.context.scene.render.fps = scene_config["fps"]
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.context.scene.render.ffmpeg.codec = "H264"
    bpy.context.scene.render.ffmpeg.constant_rate_factor = "HIGH"
    bpy.context.scene.render.ffmpeg.video_bitrate = 6000
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = "BEST"

    # Set rendering parameters
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = scene_config["number_of_samples"]
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.transparent_min_bounces = scene_config[
        "min_number_of_bounces"
    ]
    bpy.context.scene.cycles.transparent_max_bounces = scene_config[
        "max_number_of_bounces"
    ]
    bpy.context.scene.sync_mode = "AUDIO_SYNC"

    # Set render device
    if scene_config["device"] != "CPU":
        bpy.context.scene.cycles.device = "GPU"

        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = scene_config[
                "device"
            ]
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
                scene_config["device"]
            )
            
            bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            print("available devices",bpy.context.preferences.addons['cycles'].preferences.devices.keys())
            
            selected_device = ""
            for d in bpy.context.preferences.addons['cycles'].preferences.devices.keys():
                if "nvidia" in d.lower() or "tesla" in d.lower():
                    selected_device = d
                    print("Using GPU device:", selected_device)
                    break
            if selected_device == "":
                print("No GPU device found, using CPU:", list(bpy.context.preferences.addons['cycles'].preferences.devices.keys())[0])
                selected_device = list(bpy.context.preferences.addons['cycles'].preferences.devices.keys())[0]
            bpy.context.preferences.addons['cycles'].preferences.devices[selected_device].use = True



def apply_jitter(scene_config: dict[str, Any]) -> None:
    """Applies jitter (random position noise) to camera and lighting.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    add_jitter = lambda jitter: 2.0 * jitter * (np.random.random() - 0.5)

    for i in range(3):
        bpy.data.objects["Camera"].location[i] += add_jitter(
            scene_config["camera_jitter"]
        )
        bpy.data.objects["Lamp_Key"].location[i] += add_jitter(
            scene_config["key_light_jitter"]
        )
        bpy.data.objects["Lamp_Back"].location[i] += add_jitter(
            scene_config["back_light_jitter"]
        )
        bpy.data.objects["Lamp_Fill"].location[i] += add_jitter(
            scene_config["fill_light_jitter"]
        )


def set_interpolation_to_linear(bpy_obj):
    """Sets the interpolation type for all keyframes of an object to be linear.

    Args:
        bpy_obj: The blender object data
    """

    if bpy_obj.animation_data:
        for fcurve in bpy_obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def apply_recolor_cycle(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies a color change cycle to an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    # Get material nodes to insert changes
    material = bpy.data.materials[bpy_obj.data.materials[0].name]
    bsdf_node = material.node_tree.nodes["Group"]
    color_node = bsdf_node.inputs["Color"]

    # Insert periodic keyframes for color
    for state in obj["cycles"]["recolor"]["states"]:
        color_node.default_value = scene_config["colors"][state["color"]]
        color_node.keyframe_insert(data_path="default_value", frame=state["frame"])


def apply_rotate_cycle(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies a rotation cycle to an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    # Insert periodic keyframes for color
    base_rotation = bpy_obj.rotation_euler.copy()
    for state in obj["cycles"]["rotate"]["states"]:
        bpy_obj.rotation_euler = (
            base_rotation[0] + state["rotation"][0],
            base_rotation[1] + state["rotation"][1],
            base_rotation[2] + state["rotation"][2],
        )
        bpy_obj.keyframe_insert(data_path="rotation_euler", frame=state["frame"])


def apply_resize_cycle(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies a resize cycle to an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    # Insert periodic keyframes for size
    size = scene_config["sizes"][obj["size"]]
    for state in obj["cycles"]["resize"]["states"]:
        bpy.ops.transform.resize(
            value=(state["factor"], state["factor"], state["factor"])
        )
        if state["factor"] < 1.0:
            bpy.ops.transform.translate(value=(0.0, 0.0, state["factor"] * size - size))
            last_offset = state["factor"] * size - size
        elif state["factor"] > 1.0:
            bpy.ops.transform.translate(value=(0.0, 0.0, -last_offset))

        bpy_obj.keyframe_insert(data_path="scale", frame=state["frame"])
        bpy_obj.keyframe_insert(data_path="location", index=2, frame=state["frame"])


def apply_linear_cycle(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies a linear motion cycle to an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    # Insert periodic keyframes for location
    for state in obj["cycles"]["linear"]["states"]:
        last_position = bpy_obj.location.copy()
        bpy.ops.transform.translate(
            value=(
                state["location"]["x"] - last_position[0],
                state["location"]["y"] - last_position[1],
                0.0,
            )
        )
        bpy_obj.keyframe_insert(data_path="location", index=0, frame=state["frame"])
        bpy_obj.keyframe_insert(data_path="location", index=1, frame=state["frame"])


def apply_orbit_cycle(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies a linear motion cycle to an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    # Insert periodic keyframes for location
    for state in obj["cycles"]["orbit"]["states"]:
        last_position = bpy_obj.location.copy()
        bpy.ops.transform.translate(
            value=(
                state["location"]["x"] - last_position[0],
                state["location"]["y"] - last_position[1],
                0.0,
            )
        )
        bpy_obj.keyframe_insert(data_path="location", index=0, frame=state["frame"])
        bpy_obj.keyframe_insert(data_path="location", index=1, frame=state["frame"])


def apply_light_cycle(scene_config: dict[str, Any]) -> None:
    """Applies varying light intensity throughout the scene.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    # Store initial power levels to set intensity over time relative to origin
    original_key_power = bpy.data.objects["Lamp_Key"].data.energy
    original_back_power = bpy.data.objects["Lamp_Back"].data.energy
    original_fill_power = bpy.data.objects["Lamp_Fill"].data.energy
    original_area_power = bpy.data.objects["Area"].data.energy

    # Insert periodic keyframes for location
    for state in scene_config["lights"]["states"]:
        bpy.data.objects["Lamp_Key"].data.energy = original_key_power * state["intensity"]
        bpy.data.objects["Lamp_Key"].data.keyframe_insert("energy", frame=state["frame"])
        bpy.data.objects["Lamp_Back"].data.energy = original_back_power * state["intensity"]
        bpy.data.objects["Lamp_Back"].data.keyframe_insert("energy", frame=state["frame"])
        bpy.data.objects["Lamp_Fill"].data.energy = original_fill_power * state["intensity"]
        bpy.data.objects["Lamp_Fill"].data.keyframe_insert("energy", frame=state["frame"])
        bpy.data.objects["Area"].data.energy = original_area_power * state["intensity"]
        bpy.data.objects["Area"].data.keyframe_insert("energy", frame=state["frame"])


def apply_cycles(bpy_obj, obj: dict[str, Any]) -> None:
    """Applies all cycle of an object.

    Args:
        bpy_obj: The blender object data
        obj: The CycliST object data as dict
    """

    if "recolor" in obj["cycles"]:
        apply_recolor_cycle(bpy_obj, obj)
    if "rotate" in obj["cycles"]:
        apply_rotate_cycle(bpy_obj, obj)
    if "resize" in obj["cycles"]:
        apply_resize_cycle(bpy_obj, obj)
    if "linear" in obj["cycles"]:
        apply_linear_cycle(bpy_obj, obj)
    if "orbit" in obj["cycles"]:
        apply_orbit_cycle(bpy_obj, obj)


def add_objects(scene_config: dict[str, Any]) -> None:
    """Adds all the scene's objects, e.g., their positions, keyframes and materials.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    for i, obj in enumerate(scene_config["objects"]):
        # Insert object mesh into scene
        filename = os.path.join(
            scene_config["mesh_directory"],
            f"{obj['mesh']}.blend",
            "Object",
            obj["mesh"],
        )
        bpy.ops.wm.append(filename=filename)

        # Ensure unique name
        name = f"{obj['mesh']}_{i}"
        bpy_obj = bpy.data.objects[obj["mesh"]]
        bpy_obj.name = name

        # Set its initial orientation
        bpy.context.view_layer.objects.active = bpy_obj
        bpy.context.object.rotation_euler[2] = obj["angle"]

        # Set its initial location
        size = scene_config["sizes"][obj["size"]]
        bpy.ops.transform.resize(value=(size, size, size))
        bpy.ops.transform.translate(
            value=(obj["location"]["x"], obj["location"]["y"], size)
        )

        # Move a bit further up if the object is rotating
        if "cycles" in obj.keys() and "rotate" in obj["cycles"].keys():
            bpy.ops.transform.translate(value=(0.0, 0.0, size * 0.25))

        # Set its initial material and color
        material = bpy.data.materials.new(name=f"{name}_material")
        material.use_nodes = True
        bpy_obj.data.materials.append(material)

        # Set the materials nodes default color and output edge
        group_node = material.node_tree.nodes.new("ShaderNodeGroup")
        group_node.node_tree = bpy.data.node_groups[obj["material"]]

        for node in group_node.inputs:
            if node.name == "Color":
                node.default_value = scene_config["colors"][obj["color"]]
                break

        for node in material.node_tree.nodes:
            if node.name == "Material Output":
                material.node_tree.links.new(
                    group_node.outputs["Shader"],
                    node.inputs["Surface"],
                )
                break

        # Apply cycles and ensure interpolation is linear
        if "cycles" in obj.keys():
            apply_cycles(bpy_obj, obj)
        set_interpolation_to_linear(bpy_obj)


def compute_directions(scene_config: dict[str, Any]) -> None:
    """Computes the directions (behind, left, up) as vectors in this scene.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    # Compute Camera direction vectors
    camera = bpy.data.objects["Camera"]
    camera_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    camera_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    camera_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))

    # Compute Plane direction vectors relative to Camera
    behind = (camera_behind - camera_behind.project(Vector((1, 0, 0)))).normalized()
    left = (camera_left - camera_left.project(Vector((0, 1, 0)))).normalized()
    up = camera_up.project(Vector((0, 0, 1))).normalized()

    return behind, left, up


def render(scene_config: dict[str, Any]) -> None:
    """Renders a scene using Blender.

    Args:
        scene_config: The configuration of the scene following the arguments
            documented in cyclist/utility/parse_scene_config
    """

    # Get direction vectors in this scene
    behind, left, up = compute_directions(scene_config)
    scene_config["directions"] = {}
    scene_config["directions"]["behind"] = tuple(behind)
    scene_config["directions"]["front"] = tuple(-behind)
    scene_config["directions"]["left"] = tuple(left)
    scene_config["directions"]["right"] = tuple(-left)
    scene_config["directions"]["above"] = tuple(up)
    scene_config["directions"]["below"] = tuple(-up)

    # Setup scene for rendering
    setup_rendering(scene_config)

    # Setup lighting
    apply_jitter(scene_config)
    if scene_config["cyclic_lights"]:
        apply_light_cycle(scene_config)

    # Setup objects
    if "objects" in scene_config.keys():
        add_objects(
            scene_config
        )

    # Render video with Blender
    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all()
    start = time()
    bpy.ops.render.render(animation=True)
    scene_config["render_time"] = time() - start
    scene_config["rendered"] = True

    # Optionally store .blend file
    if scene_config["write_blendfile"]:
        blend_file_path = os.path.join(
            scene_config["blendfile_path"], scene_config["blend_file"]
        )
        bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)

    # Write-out pre-rendering scene configuration
    # We have to do it here since it happens in a separate process from cyclist.py
    scene_config_file_path = os.path.join(
        scene_config["scene_config_directory"],
        scene_config["scene_config_file"],
    )
    with open(scene_config_file_path, "w") as scene_config_file:
        json.dump(scene_config, scene_config_file, indent=2)


if __name__ == "__main__":
    # Check if running in Blender environment and getting scene_config
    try:
        import sys
        import bpy, bpy_extras
        from mathutils import Vector

        # Inside blender, we only consider the scene configuration path after '-- --scene_config '
        scene_config_path = sys.argv[-1].split(" ")[-1]
    except ImportError:
        print(
            "Running cyclist/scene/renderer.py as main file outside of Blender does not work."
        )
        exit(1)

    # Render scene as described in config and end
    with open(scene_config_path, "r") as scene_config_file:
        scene_config = json.load(scene_config_file)
        render(scene_config)
