#!/usr/bin/env python3
"""
Blender utility helpers.

This module provides:
  • Headless Blender smoke tests and sample renders (run via system Python).
  • Scene utilities callable from inside Blender (e.g. rendering asset bundles).
"""

import logging
import argparse
import json
import os
import os.path as osp
import pickle
import subprocess
import sys
from glob import glob
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
COLOR_CHOICES: Dict[str, Tuple[float, float, float]] = {
    "green": (0.45, 0.75, 0.55),
    "lightgreen": (0.68, 0.85, 0.60),
    "purple": (0.67, 0.57, 0.86),
    "red": (0.93, 0.45, 0.45),
    "yellow": (0.96, 0.88, 0.55),
    "blue1": (0.45, 0.63, 0.95),
    "blue2": (0.60, 0.75, 0.98),
    "pink": (0.94, 0.66, 0.86),
    "darkgray": (0.25, 0.25, 0.25),
}

DEFAULT_HAND_COLORS: Tuple[str, str] = ("blue1", "blue2")
DEFAULT_OBJECT_COLOR: str = "pink"

_HAND_COLOR_SELECTION: Tuple[str, str] = DEFAULT_HAND_COLORS
_OBJECT_COLOR_SELECTION: str = DEFAULT_OBJECT_COLOR
_VIS_CONTACT: bool = False
_RENDER_HAND: bool = True
_RENDER_OBJ_TRAIL: bool = False


def _validate_color_name(name: str) -> str:
    key = name.lower()
    if key not in COLOR_CHOICES:
        raise ValueError(
            f"Unknown color '{name}'. Available options: {', '.join(sorted(COLOR_CHOICES))}"
        )
    return key


def _parse_hand_colors(option: str) -> Tuple[str, str]:
    parts = [p.strip() for p in option.split(",") if p.strip()]
    if not parts:
        parts = list(DEFAULT_HAND_COLORS)
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    elif len(parts) > 2:
        parts = parts[:2]
    return (_validate_color_name(parts[0]), _validate_color_name(parts[1]))


def _apply_contact_color(mesh_info: Dict) -> Dict:
    if not _VIS_CONTACT:
        return mesh_info
    color = mesh_info.get("color")
    if color is None:
        return mesh_info
    color_arr = np.asarray(color, dtype=np.float32)
    if color_arr.size < 3:
        return mesh_info
    if np.allclose(color_arr[:3], 1.0, atol=1e-3):
        red = COLOR_CHOICES["red"]
        alpha = float(color_arr[3]) if color_arr.size >= 4 else 1.0
        mesh_info["color"] = [red[0], red[1], red[2], alpha]
        mesh_info["force_color"] = True
        mesh_info["alpha"] = alpha
    return mesh_info

try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except ImportError:  # pragma: no cover - bpy only available inside Blender
    bpy = None  # type: ignore
    Matrix = None  # type: ignore
    Vector = None  # type: ignore

LOGGER = logging.getLogger("mayday.blender_vis")


def _ensure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _run(
    cmd: Sequence[str],
    description: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    print(f"\n[blender-test] {description}\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({description}) with exit code {result.returncode}"
        )
    return result


def _list_asset_files(asset_root: str) -> List[str]:
    asset_root = osp.abspath(asset_root)
    pkl_files = sorted(
        path
        for path in glob(osp.join(asset_root, "*.pkl"))
        if osp.isfile(path)
    )
    if pkl_files:
        return pkl_files

    dirs = [
        osp.join(asset_root, entry)
        for entry in sorted(os.listdir(asset_root))
        if len(entry) == 4 and entry.isdigit() and osp.isdir(osp.join(asset_root, entry))
    ]
    return dirs


def _load_asset_bundle(path: str) -> Dict:
    if osp.isdir(path):
        candidate = osp.join(path, "assets.pkl")
        if not osp.isfile(candidate):
            raise FileNotFoundError(
                f"No assets.pkl found under directory '{path}'. "
                "If you migrated to flat .pkl layout, point render_assets_with_blender "
                "to the new directory."
            )
        path = candidate
    with open(path, "rb") as f:
        return pickle.load(f)


def _clear_scene():
    if bpy is None:
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)


def _ensure_material(name: str, color: Iterable[float], alpha: float = 1.0):
    if bpy is None:
        return None

    mat_name = f"{name}_mat"
    material = bpy.data.materials.get(mat_name)
    if material is None:
        material = bpy.data.materials.new(mat_name)
        material.use_nodes = True

    material.blend_method = "BLEND"
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (
        float(color[0]),
        float(color[1]),
        float(color[2]),
        1.0,
    )
    principled.inputs["Alpha"].default_value = float(alpha)
    principled.inputs["Roughness"].default_value = 0.5
    principled.inputs["Metallic"].default_value = 0.0
    return material


def _mesh_color_palette(name: str, base_color: Iterable[float]) -> Tuple[float, float, float]:
    name_lower = name.lower()
    if "left" in name_lower:
        return COLOR_CHOICES[_HAND_COLOR_SELECTION[0]]
    if "right" in name_lower:
        return COLOR_CHOICES[_HAND_COLOR_SELECTION[1]]
    if "object" in name_lower:
        return COLOR_CHOICES[_OBJECT_COLOR_SELECTION]
    return tuple(float(c) for c in base_color[:3])


hand_metallic = 0.05
# hand_metallic = 0.1
def _material_params(name: str) -> Tuple[float, float]:
    name_lower = name.lower()
    if "left" in name_lower or "right" in name_lower:
        return 0.55, hand_metallic
    if "object" in name_lower:
        return 0.35, 0.0
    return 0.6, 0.0


def _build_mesh(mesh_info: Dict, stylized: bool = True):
    if bpy is None:
        raise RuntimeError("Blender (bpy) is required to create meshes.")

    name = mesh_info["name"]
    verts = mesh_info["vertices"]
    faces = mesh_info["faces"]
    color = mesh_info.get("color", [0.8, 0.8, 0.8, 1.0])
    alpha = mesh_info.get("alpha", color[-1] if len(color) == 4 else 1.0)

    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(
        [tuple(map(float, v)) for v in verts],
        [],
        [tuple(map(int, f)) for f in faces],
    )
    mesh_data.update(calc_edges=True)
    for poly in mesh_data.polygons:
        poly.use_smooth = True

    mesh_obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    metallic = 0.
    force_color = bool(mesh_info.get("force_color"))
    if stylized and not force_color:
        palette_color = _mesh_color_palette(name, color)
        roughness, metallic = _material_params(name)
        _assign_material(mesh_obj, palette_color, roughness=roughness, metallic=metallic, alpha=alpha)
    else:
        force_palette = tuple(color[:3]) if len(color) >= 3 else (0.8, 0.8, 0.8)
        _assign_material(mesh_obj, force_palette, roughness=0.3, metallic=metallic, alpha=alpha)
    return mesh_obj


def _choose_render_engine(scene) -> str:
    if bpy is None:
        raise RuntimeError("Blender (bpy) is required to configure render engines.")
    eng_prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
    available = [item.identifier for item in eng_prop.enum_items]
    for preferred in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if preferred in available:
            scene.render.engine = preferred
            return preferred
    if not available:
        raise RuntimeError("No render engines available in this Blender build.")
    fallback = available[0]
    scene.render.engine = fallback
    return fallback


def pytorch3d_to_blender_camera(wTc_pytorch3d):
    """
    Convert camera-to-world transform from PyTorch3D to Blender convention.
    
    PyTorch3D convention:
    - World: +Y up, +X left, +Z forward (into scene)
    - Camera looks along +Z
    
    Blender convention:
    - World: +Z up, +X right, +Y forward
    - Camera looks along -Z (local)
    
    Args:
        wTc_pytorch3d: (4, 4) camera-to-world matrix in PyTorch3D convention
        
    Returns:
        wTc_blender: (4, 4) camera-to-world matrix in Blender convention
    """
    camera_rotation = np.array([
        [ 1,  0,  0,  0],  # Keep X
        [ 0, -1,  0,  0],  # Flip Y: -Y up -> +Y up
        [ 0,  0, -1,  0],  # Flip Z: +Z front -> -Z front
        [ 0,  0,  0,  1]
    ])
    
    # camera_rotation = np.array([
    #     [-1,  0,  0,  0],  # Flip X: left -> right
    #     [ 0,  1,  0,  0],  # Keep Y: up -> up
    #     [ 0,  0, -1,  0],  # Flip Z: forward -> backward
    #     [ 0,  0,  0,  1]
    # ])
    
    wTc_blender = wTc_pytorch3d @ camera_rotation

    return wTc_blender


def _configure_camera_from_assets(camera_assets: Dict) -> Tuple[object, object]:
    if bpy is None:
        raise RuntimeError("Blender (bpy) is required to create cameras.")

    width = int(camera_assets["width"])
    height = int(camera_assets["height"])
    K = np.asarray(camera_assets["intrinsic"], dtype=np.float32)
    wTc_pytorch3d = np.asarray(camera_assets["extrinsic_wTc"], dtype=np.float32)

    wTc_blender = pytorch3d_to_blender_camera(wTc_pytorch3d)

    cam_data = bpy.data.cameras.new("PredictedCamera")
    cam_obj = bpy.data.objects.new("PredictedCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    sensor_width = 36.0
    sensor_height = sensor_width * height / max(width, 1)
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_height
    cam_data.lens = sensor_width * K[0, 0] / max(width, 1)
    cam_data.lens_unit = "MILLIMETERS"

    cam_data.shift_x = (K[0, 2] - width / 2) / width
    cam_data.shift_y = (height / 2 - K[1, 2]) / height


    cam_obj.matrix_world = Matrix([tuple(row) for row in wTc_blender])

    scene = bpy.context.scene
    _choose_render_engine(scene)
    scene.camera = cam_obj
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    return cam_obj, cam_data


def _look_at(camera_obj, target: Tuple[float, float, float]):
    if bpy is None or Vector is None:
        return
    direction = Vector(target) - camera_obj.location
    direction.normalize()
    rotation = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = rotation


def _assign_material(obj, color, roughness=0.45, metallic=0.0, alpha=1.0):
    material = _ensure_material(obj.name, color, alpha)
    if material:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = metallic
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
def _grid_floor_material(
    name: str = "FloorGrid",
    base_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    line_color: Tuple[float, float, float] = (0.85, 0.85, 0.85),
    scale: float = 20.0,
    line_width: float = 0.015,
):
    """Create a pure white floor material with gray grid lines."""
    if bpy is None:
        return None
    
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    # Output and Principled BSDF
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.0

    # Texture coordinate and mapping
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, 1.0)

    # Separate XYZ
    separate = nodes.new("ShaderNodeSeparateXYZ")
    
    # X direction processing
    fract_x = nodes.new("ShaderNodeMath")
    fract_x.operation = "FRACT"
    fract_x.label = "Fract X"
    
    sub_x = nodes.new("ShaderNodeMath")
    sub_x.operation = "SUBTRACT"
    sub_x.inputs[1].default_value = 0.5
    
    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    
    compare_x = nodes.new("ShaderNodeMath")
    compare_x.operation = "GREATER_THAN"
    compare_x.inputs[1].default_value = 0.5 - line_width
    
    # Y direction processing
    fract_y = nodes.new("ShaderNodeMath")
    fract_y.operation = "FRACT"
    fract_y.label = "Fract Y"
    
    sub_y = nodes.new("ShaderNodeMath")
    sub_y.operation = "SUBTRACT"
    sub_y.inputs[1].default_value = 0.5
    
    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = "ABSOLUTE"
    
    compare_y = nodes.new("ShaderNodeMath")
    compare_y.operation = "GREATER_THAN"
    compare_y.inputs[1].default_value = 0.5 - line_width
    
    # Combine X and Y
    combine = nodes.new("ShaderNodeMath")
    combine.operation = "MAXIMUM"
    
    # Mix colors
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.inputs[6].default_value = (*base_color, 1.0)  # A socket - white
    mix.inputs[7].default_value = (*line_color, 1.0)  # B socket - gray lines

    # Connect everything
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], separate.inputs["Vector"])
    
    # X chain
    links.new(separate.outputs["X"], fract_x.inputs[0])
    links.new(fract_x.outputs[0], sub_x.inputs[0])
    links.new(sub_x.outputs[0], abs_x.inputs[0])
    links.new(abs_x.outputs[0], compare_x.inputs[0])
    
    # Y chain
    links.new(separate.outputs["Y"], fract_y.inputs[0])
    links.new(fract_y.outputs[0], sub_y.inputs[0])
    links.new(sub_y.outputs[0], abs_y.inputs[0])
    links.new(abs_y.outputs[0], compare_y.inputs[0])
    
    # Combine and output
    links.new(compare_x.outputs[0], combine.inputs[0])
    links.new(compare_y.outputs[0], combine.inputs[1])
    links.new(combine.outputs[0], mix.inputs[0])  # Factor
    links.new(mix.outputs[2], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return material


def _create_trail(name: str, points: List[Sequence[float]], color: Tuple[float, float, float], width: float = 0.01):
    print("CREATE TRAIL?????", name)
    if bpy is None or len(points) < 2:
        return
    curve = bpy.data.curves.new(name=f"{name}_trail_curve", type="CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new(type="POLY")
    spline.points.add(len(points) - 1)
    for idx, pt in enumerate(points):
        x, y, z = pt
        spline.points[idx].co = (float(x), float(y), float(z), 1.0)
    curve.bevel_depth = width
    curve.bevel_resolution = 2
    curve_obj = bpy.data.objects.new(f"{name}_trail", curve)
    bpy.context.collection.objects.link(curve_obj)
    _assign_material(curve_obj, color, roughness=0.4, metallic=0.0)

def _create_camera_wireframe(name: str, mat: np.ndarray, size: float = 0.05):
    if bpy is None:
        return None
    
    # Define the pyramid vertices
    apex = (0.0, 0.0, 0.0)
    corners = [
        (-size, -size, -size * 1.5),
        (size, -size, -size * 1.5),
        (size, size, -size * 1.5),
        (-size, size, -size * 1.5),
    ]
    
    # Create a curve for each edge
    curve_data = bpy.data.curves.new(name=f"{name}_cam_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.002  # Thickness of the wireframe lines
    curve_data.bevel_resolution = 2
    
    # Edges from apex to corners
    for corner in corners:
        spline = curve_data.splines.new(type='POLY')
        spline.points.add(1)  # Add one more point (total 2)
        spline.points[0].co = (*apex, 1.0)
        spline.points[1].co = (*corner, 1.0)
    
    # Edges around the base
    for i in range(4):
        spline = curve_data.splines.new(type='POLY')
        spline.points.add(1)
        spline.points[0].co = (*corners[i], 1.0)
        spline.points[1].co = (*corners[(i + 1) % 4], 1.0)
    
    obj = bpy.data.objects.new(f"{name}_camera", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = Matrix([tuple(row) for row in mat])
    
    # Assign material for proper rendering
    _assign_material(obj, COLOR_CHOICES["darkgray"], roughness=0.6, metallic=0.0)
    
    return obj

def _setup_environment(
    scene,
    include_floor: bool = True,
    floor_center: Tuple[float, float, float] | None = None,
):
    # FLOOR: z= -1
    if bpy is None:
        return

    # Bright, neutral ambient light
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    # bg.inputs["Color"].default_value = (0.98, 0.98, 0.98, 1.0)
    bg.inputs["Color"].default_value = (1, 1, 1, 1.0)
    bg.inputs["Strength"].default_value = 0.35

    if include_floor:
        if floor_center is not None:
            floor_x, floor_y, floor_z = (
                float(floor_center[0]),
                float(floor_center[1]),
                float(floor_center[2]),
            )
        else:
            floor_x, floor_y, floor_z = 0.0, 0.0, -1.0

        # Floor plane - larger and properly scaled
        bpy.ops.mesh.primitive_plane_add(size=2.5, scale=(0.5, 1., 1.))  # wide table
        floor = bpy.context.object
        floor.name = "Floor"
        
        floor.scale = (1., .5, 1.0)  # This will persist when you save/reopen

        # Apply grid material
        grid_material = _grid_floor_material(
            # base_color=(1.0, 1.0, 1.0),  # Pure white
            base_color=(30., 30., 30.),  # overwhite
            line_color=(0.88, 0.88, 0.88),  # Light gray lines
            scale=3.0,  # Grid density
            line_width=0.015  # Line thickness
        )
        if grid_material:
            if floor.data.materials:
                floor.data.materials[0] = grid_material
            else:
                floor.data.materials.append(grid_material)
        
        floor.location = (floor_x, floor_y, floor_z)

    # # Ambient light
    # scene.world.use_nodes = True
    # bg = scene.world.node_tree.nodes["Background"]
    # bg.inputs["Color"].default_value = (0.70, 0.70, 0.72, 1.0)
    # bg.inputs["Strength"].default_value = 0.2

    # if include_floor:
    #     # Floor plane
    #     bpy.ops.mesh.primitive_plane_add(size=60)
    #     floor = bpy.context.object
    #     floor.name = "Floor"
    #     grid_material = _grid_floor_material()
    #     if grid_material:
    #         if floor.data.materials:
    #             floor.data.materials[0] = grid_material
    #         else:
    #             floor.data.materials.append(grid_material)
    #     else:
    #         _assign_material(floor, (0.96, 0.96, 0.96), roughness=0.85)
    #     floor.location = (0.0, 0.0, -1)

    #     # Backdrop wall
    #     bpy.ops.mesh.primitive_plane_add(size=60)
    #     wall = bpy.context.object
    #     wall.name = "Backdrop"
    #     wall.rotation_euler = (1.5708, 0.0, 0.0)
    #     wall.location = (0.0, -30.0, 30.0)
    #     _assign_material(wall, (0.78, 0.78, 0.80), roughness=0.9)

    # Key light
    # key_data = bpy.data.lights.new("KeyLight", type="AREA")
    # key_data.energy = 1200
    # key_data.color = (1.0, 0.92, 0.82)
    # key_data.size = 7
    # key = bpy.data.objects.new("KeyLight", key_data)
    # key.location = (6.0, -6.0, 8.0)
    # key.rotation_euler = (1.1, 0.0, 0.8)
    # bpy.context.collection.objects.link(key)


    # Main directional key light for sharp shadows - pointing downward
    key_data = bpy.data.lights.new("KeyLight", type="SUN")  # SUN for sharp parallel shadows
    key_data.energy = 1.6  # Reduce brightness
    key_data.angle = 0.03  # Slightly softer shadows
    key = bpy.data.objects.new("KeyLight", key_data)
    key.location = (5.0, -5.0, 10.0)  # Position above the scene
    key.rotation_euler = (0.0, 0.0, 0.0)  # Point straight down (-Z direction)
    bpy.context.collection.objects.link(key)
    
    # Fill light
    fill_data = bpy.data.lights.new("FillLight", type="AREA")
    fill_data.energy = 220
    fill_data.color = (0.78, 0.86, 1.0)
    fill_data.size = 6
    fill = bpy.data.objects.new("FillLight", fill_data)
    fill.location = (-6.0, -3.0, 1.0)
    fill.rotation_euler = (1.2, 0.0, -0.7)
    bpy.context.collection.objects.link(fill)

    # Rim light
    rim_data = bpy.data.lights.new("RimLight", type="SPOT")
    rim_data.energy = 280
    rim_data.color = (1.0, 0.95, 0.90)
    rim_data.spot_size = 1.0
    rim_data.shadow_soft_size = 1.5
    rim = bpy.data.objects.new("RimLight", rim_data)
    rim.location = (0.0, 0.0, 10.0)
    rim.rotation_euler = (1.4, 0.0, 0.0)
    bpy.context.collection.objects.link(rim)

    # Overhead soft fill
    top_data = bpy.data.lights.new("TopFill", type="AREA")
    top_data.energy = 220
    top_data.size = 10
    top_data.color = (1.0, 0.96, 0.90)
    top = bpy.data.objects.new("TopFill", top_data)
    top.location = (0.0, 0.0, 9.0)
    top.rotation_euler = (3.1416, 0.0, 0.0)
    bpy.context.collection.objects.link(top)

    # Overhead light
    top_data = bpy.data.lights.new("TopFill", type="AREA")
    top_data.energy = 320
    top_data.size = 10
    top_data.color = (1.0, 0.96, 0.90)
    top = bpy.data.objects.new("TopFill", top_data)
    top.location = (0.0, 0.0, 9.0)
    top.rotation_euler = (3.1416, 0.0, 0.0)
    bpy.context.collection.objects.link(top)


def _configure_eevee(scene):
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        return
    if hasattr(eevee, "use_soft_shadows"):
        eevee.use_soft_shadows = True
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = True
        if hasattr(eevee, "gtao_distance"):
            eevee.gtao_distance = 0.8
        if hasattr(eevee, "gtao_factor"):
            eevee.gtao_factor = 1.0
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = True
        if hasattr(eevee, "bloom_threshold"):
            eevee.bloom_threshold = 1.0
        if hasattr(eevee, "bloom_intensity"):
            eevee.bloom_intensity = 0.05
    view_settings = getattr(scene, "view_settings", None)
    if view_settings:
        view_settings.view_transform = "Filmic"
        view_settings.look = "High Contrast"
        view_settings.exposure = -0.6
        view_settings.gamma = 1.0


def render_assets_with_blender(
    asset_dir: str,
    output_dir: str,
    target_frame: int = 50,
    allocentric_step: int = 50,
    allocentric_frames: Sequence[int] | None = None,
    external_camera_location: Tuple[float, float, float] = (1.5, -1.5, 1.0),
    external_camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.3),
    blend_save_path: str | None = None,
    hand_colors: Tuple[str, str] = ("blue1", "blue2"),
    object_color: str = "pink",
    render_camera: bool = False,
    render_trail: bool = True,
    render_allocentric: bool = True,
    render_target_frame: bool = True,
    render_hand: bool = True,
    render_obj_trail: bool = False,
    vis_contact: bool = False,
) -> None:
    if bpy is None:
        raise RuntimeError(
            "render_assets_with_blender must be executed within Blender's Python "
            "environment (bpy is unavailable)."
        )

    _ensure_logger()
    global _HAND_COLOR_SELECTION, _OBJECT_COLOR_SELECTION, _VIS_CONTACT
    _HAND_COLOR_SELECTION = hand_colors
    _OBJECT_COLOR_SELECTION = object_color
    _VIS_CONTACT = vis_contact
    global _RENDER_HAND, _RENDER_OBJ_TRAIL
    _RENDER_HAND = render_hand
    _RENDER_OBJ_TRAIL = render_obj_trail
    asset_dir = osp.abspath(asset_dir)
    output_dir = osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frame_assets = _list_asset_files(asset_dir)
    if not frame_assets:
        raise FileNotFoundError(f"No asset bundles found under {asset_dir}")

    scene = bpy.context.scene

    target_idx = min(target_frame, len(frame_assets) - 1)
    frame_data = _load_asset_bundle(frame_assets[target_idx])
    alloc_camera_info = frame_data.get("alloc_camera")
    if render_target_frame:
        LOGGER.info(
            "Rendering predicted camera view for frame %d (using %s)",
            target_idx,
            frame_assets[target_idx],
        )

        _clear_scene()
        _setup_environment(scene, include_floor=False)
        _choose_render_engine(scene)
        _configure_eevee(scene)

        for mesh_info in frame_data["meshes"]:
            if not _RENDER_HAND and "hand" in mesh_info.get("name", "").lower():
                continue
            _build_mesh(_apply_contact_color(dict(mesh_info)))
        _configure_camera_from_assets(frame_data["camera"])
        input_frame = frame_data.get("image_path")
        if input_frame and osp.exists(input_frame):
            try:
                Image.open(input_frame).save(osp.join(output_dir, f"{target_idx:04d}_input.png"))
            except Exception as exc:
                LOGGER.warning("Failed to save input frame %s: %s", input_frame, exc)

        pred_output_path = osp.join(output_dir, f"{target_idx:04d}_camera.png")
        bpy.context.scene.render.filepath = pred_output_path
        bpy.ops.render.render(write_still=True)
        LOGGER.info("Saved predicted camera render to %s", pred_output_path)
    else:
        LOGGER.info("Skipping predicted camera render (--no-render-target-frame specified).")

    if render_allocentric:
        if allocentric_frames is not None:
            try:
                sampled_indices = sorted(dict.fromkeys(int(idx) for idx in allocentric_frames))
            except ValueError as exc:
                raise ValueError("allocentric_frames must contain integer indices") from exc
            if not sampled_indices:
                raise ValueError("allocentric_frames provided but empty after parsing")
            invalid = [idx for idx in sampled_indices if idx < 0 or idx >= len(frame_assets)]
            if invalid:
                raise ValueError(
                    f"Allocentric frame indices out of range (total {len(frame_assets)}): {invalid}"
                )
            LOGGER.info(
                "Rendering allocentric overlay using frames %s (total frames: %d)",
                sampled_indices,
                len(frame_assets),
            )
        else:
            LOGGER.info(
                "Rendering allocentric overlay sampling every %d frames (total frames: %d)",
                allocentric_step,
                len(frame_assets),
            )
            sampled_indices = list(range(0, len(frame_assets), max(allocentric_step, 1)))
            if sampled_indices[-1] != len(frame_assets) - 1:
                sampled_indices.append(len(frame_assets) - 1)


        camera_positions: List[np.ndarray] = []
        trail_points = {"left": [], "right": [], "camera": []}
        obj_trail_points: Dict[str, List[List[float]]] = {}
        obj_trail_colors: Dict[str, Tuple[float, float, float]] = {}
        for path in frame_assets:
            frame_assets_full = _load_asset_bundle(path)
            camera_info_full = frame_assets_full.get("camera")
            if camera_info_full is not None:
                wTc_full = np.asarray(camera_info_full.get("extrinsic_wTc"), dtype=np.float32)
                cam_mat_full = pytorch3d_to_blender_camera(wTc_full)
                cam_pos = cam_mat_full[:3, 3]
                camera_positions.append(cam_pos)
                if render_trail:
                    trail_points["camera"].append(cam_pos.tolist())
            for mesh_info in frame_assets_full["meshes"]:
                verts = mesh_info.get("vertices")
                if verts is None or len(verts) == 0:
                    continue
                name = mesh_info.get("name", "").lower()
                if render_trail and _RENDER_HAND:
                    if "left_hand" in name:
                        trail_points["left"].append(verts[0])
                        continue
                    if "right_hand" in name:
                        trail_points["right"].append(verts[0])
                        continue
                if _RENDER_OBJ_TRAIL and "object" in name:
                    verts_np = np.asarray(verts, dtype=np.float32)
                    if verts_np.ndim == 2 and verts_np.shape[1] == 3:
                        center = verts_np.mean(axis=0)
                        obj_name = mesh_info.get("name", "object")
                        obj_trail_points.setdefault(obj_name, []).append(center.tolist())
                        color = mesh_info.get("color")
                        if color is not None and len(color) >= 3:
                            obj_trail_colors[obj_name] = tuple(float(c) for c in color[:3])

        floor_center: Tuple[float, float, float] | None = None
        if alloc_camera_info is not None:
            alloc_wTc = np.asarray(alloc_camera_info.get("extrinsic_wTc"), dtype=np.float32)
            alloc_mat = pytorch3d_to_blender_camera(alloc_wTc)
            alloc_pos = alloc_mat[:3, 3]
            floor_center = (
                float(alloc_pos[0]),
                float(alloc_pos[1]) + 1,
                -1
            )

        _clear_scene()
        _setup_environment(scene, include_floor=True, floor_center=floor_center)
        _choose_render_engine(scene)
        _configure_eevee(scene)


        for order, idx in enumerate(sampled_indices):
            assets = _load_asset_bundle(frame_assets[idx])
            frame_factor = idx / max(len(frame_assets) - 1, 1)
            for mesh_info in assets["meshes"]:
                if not _RENDER_HAND and "hand" in mesh_info.get("name", "").lower():
                    continue
                mesh_info = _apply_contact_color(dict(mesh_info))
                mesh_info["name"] = f"{mesh_info['name']}_t{idx:04d}"
                base_color = np.array(mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]), dtype=np.float32)
                if _VIS_CONTACT and np.allclose(base_color[:3], COLOR_CHOICES["red"], atol=1e-3):
                    mesh_info["color"] = [*COLOR_CHOICES["red"], float(base_color[3] if base_color.size == 4 else 1.0)]
                else:
                    blended = np.clip(base_color * (0.6 + 0.4 * (1 - frame_factor)), 0.0, 1.0)
                    mesh_info["color"] = blended.tolist()
                mesh_info["alpha"] = 1
                _build_mesh(mesh_info)
            if render_camera:
                camera_info = assets.get("camera")
                if camera_info is not None:
                    wTc_pred = np.asarray(camera_info.get("extrinsic_wTc"), dtype=np.float32)
                    wireframe_mat = pytorch3d_to_blender_camera(wTc_pred)
                    _create_camera_wireframe(
                        f"predicted_cam_t{idx:04d}",
                        wireframe_mat,
                        size=0.05,
                    )

        if render_trail and _RENDER_HAND:
            _create_trail(
                "left_hand",
                trail_points["left"],
                COLOR_CHOICES[_HAND_COLOR_SELECTION[0]],
                width=0.002,
            )
            _create_trail(
                "right_hand",
                trail_points["right"],
                COLOR_CHOICES[_HAND_COLOR_SELECTION[1]],
                width=0.002,
            )
            if render_camera:
                _create_trail(
                    "predicted_camera",
                    trail_points["camera"],
                    COLOR_CHOICES["darkgray"],
                    width=0.0015,
                )
        if _RENDER_OBJ_TRAIL:
            for obj_name, points in obj_trail_points.items():
                if len(points) < 2:
                    continue
                color = obj_trail_colors.get(obj_name, COLOR_CHOICES[_OBJECT_COLOR_SELECTION])
                _create_trail(
                    obj_name,
                    points,
                    color,
                    width=0.0025,
                )

        scene = bpy.context.scene
        _choose_render_engine(scene)
        if alloc_camera_info is not None:
            cam_obj, _ = _configure_camera_from_assets(alloc_camera_info)
        else:
            # scene.render.resolution_x = 1920
            # scene.render.resolution_y = 1080
            # get a 4:3
            scene.render.resolution_x = 1440
            scene.render.resolution_y = 1080
            scene.render.resolution_percentage = 100
            scene.render.film_transparent = True

            alloc_cam_data = bpy.data.cameras.new("AllocentricCamera")
            alloc_cam_obj = bpy.data.objects.new("AllocentricCamera", alloc_cam_data)
            bpy.context.collection.objects.link(alloc_cam_obj)
            alloc_cam_obj.location = Vector(external_camera_location)
            _look_at(alloc_cam_obj, external_camera_target)
            alloc_cam_data.lens = 35.0
            scene.camera = alloc_cam_obj

        allocentric_path = osp.join(output_dir, "allocentric_overlay.png")
        bpy.context.scene.render.filepath = allocentric_path
        bpy.ops.render.render(write_still=True)
        LOGGER.info("Saved allocentric overlay render to %s", allocentric_path)

        if alloc_camera_info is not None:
            export_camera = {}
            for key, value in alloc_camera_info.items():
                if isinstance(value, np.ndarray):
                    export_camera[key] = value.tolist()
                else:
                    export_camera[key] = value
            camera_export_path = osp.join(output_dir, "allocentric_camera.json")
            with open(camera_export_path, "w", encoding="utf-8") as f:
                json.dump(export_camera, f, indent=2)
            LOGGER.info("Saved allocentric camera parameters to %s", camera_export_path)
    else:
        LOGGER.info("Skipping allocentric overlay render (--no-render-allocentric specified).")

    if blend_save_path:
        blend_save_path = osp.abspath(blend_save_path)
        os.makedirs(osp.dirname(blend_save_path), exist_ok=True)
        LOGGER.info("Saving Blender scene (with floor) to %s", blend_save_path)
        bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender utilities")
    parser.add_argument(
        "--mode",
        choices=["smoke", "render_sphere", "render_assets"],
        default="render_assets",
        help="Operation to perform.",
    )
    parser.add_argument(
        "--blender-root",
        help="Path to the unpacked Blender directory (e.g. blender-4.3.2-linux-x64)",
        default='/move/u/yufeiy2/Package/blender-4.3.2-linux-x64'
    )
    parser.add_argument(
        "--blender-exe",
        help="Optional explicit path to the Blender executable; "
        "defaults to <root>/blender",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for rendered images "
        "(render_sphere and render_assets modes).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (square) when rendering the sphere.",
    )
    parser.add_argument(
        "--image-name",
        default="unit_sphere.png",
        help="Filename for the rendered image.",
    )
    parser.add_argument(
        "--asset-dir",
        help="Directory containing per-frame asset pickles (render_assets mode).",
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=50,
        help="Frame index to render from the predicted camera (render_assets mode).",
    )
    parser.add_argument(
        "--allocentric-step",
        type=int,
        default=50,
        help="Stride when sampling frames for the allocentric overlay.",
    )
    parser.add_argument(
        "--allocentric-frames",
        help="Comma-separated list of frame indices to render for the allocentric overlay.",
    )
    parser.add_argument(
        "--external-camera-location",
        nargs=3,
        type=float,
        default=(1., -2., .5),
        metavar=("X", "Y", "Z"),
        help="Allocentric camera location (render_assets mode).",
    )
    parser.add_argument(
        "--external-camera-target",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.),
        metavar=("X", "Y", "Z"),
        help="Point the allocentric camera looks at (render_assets mode).",
    )
    parser.add_argument(
        "--save-blend",
        help="If provided in render_assets mode, save the constructed scene to this .blend file.",
    )
    parser.add_argument(
        "--hand-color",
        default="blue1,blue2",
        help="Comma-separated colors for left and right hands "
             f"(options: {', '.join(sorted(COLOR_CHOICES))}).",
    )
    parser.add_argument(
        "--object-color",
        default="pink",
        help=f"Object color name (options: {', '.join(sorted(COLOR_CHOICES))}).",
    )
    parser.add_argument(
        "--render-camera",
        action="store_true",
        help="If set, draw allocentric camera wireframes for sampled frames.",
    )
    parser.add_argument(
        "--render-allocentric",
        dest="render_allocentric",
        action="store_true",
        help="Render allocentric overlay for sampled frames.",
    )
    parser.add_argument(
        "--no-render-allocentric",
        dest="render_allocentric",
        action="store_false",
        help="Disable allocentric overlay render.",
    )
    parser.add_argument(
        "--render-trail",
        dest="render_trail",
        action="store_true",
        help="Draw hand and camera motion trails in the allocentric view.",
    )
    parser.add_argument(
        "--no-render-trail",
        dest="render_trail",
        action="store_false",
        help="Disable hand and camera motion trails.",
    )
    parser.add_argument(
        "--render-target-frame",
        dest="render_target_frame",
        action="store_true",
        help="Render the predicted camera view for the target frame.",
    )
    parser.add_argument(
        "--no-render-target-frame",
        dest="render_target_frame",
        action="store_false",
        help="Disable predicted camera rendering for the target frame.",
    )
    parser.add_argument(
        "--render-hand",
        dest="render_hand",
        action="store_true",
        help="Render hand meshes.",
    )
    parser.add_argument(
        "--no-render-hand",
        dest="render_hand",
        action="store_false",
        help="Skip rendering hand meshes.",
    )
    parser.add_argument(
        "--vis-obj-trail",
        dest="render_obj_trail",
        action="store_true",
        help="Visualize object center-of-mass trails.",
    )
    parser.add_argument(
        "--no-vis-obj-trail",
        dest="render_obj_trail",
        action="store_false",
        help="Disable object center-of-mass trails.",
    )
    parser.add_argument(
        "--vis-contact",
        dest="vis_contact",
        action="store_true",
        help="Highlight contact vertices using the red palette color.",
    )
    parser.add_argument(
        "--no-vis-contact",
        dest="vis_contact",
        action="store_false",
        help="Disable contact highlighting.",
    )
    parser.set_defaults(
        render_trail=True,
        render_allocentric=True,
        render_target_frame=True,
        render_hand=True,
        render_obj_trail=False,
        vis_contact=False,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        raw_argv = sys.argv[1:]
        if "--" in raw_argv:
            idx = raw_argv.index("--")
            raw_argv = raw_argv[idx + 1 :]
        args = parse_args(raw_argv)
    else:
        args = parse_args(argv)
    if args.mode == "render_assets":
        if not args.output_dir:
            raise ValueError("--output-dir is required when --mode=render_assets")
        if not args.asset_dir:
            raise ValueError("--asset-dir is required when --mode=render_assets")
        try:
            hand_colors = _parse_hand_colors(args.hand_color)
            object_color = _validate_color_name(args.object_color)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        allocentric_frames: List[int] | None = None
        if args.allocentric_frames:
            tokens = [token.strip() for token in args.allocentric_frames.split(",") if token.strip()]
            if not tokens:
                raise SystemExit("--allocentric-frames provided but empty")
            try:
                allocentric_frames = [int(token) for token in tokens]
            except ValueError as exc:
                raise SystemExit("--allocentric-frames must contain integers") from exc

        render_assets_with_blender(
            asset_dir=args.asset_dir,
            output_dir=args.output_dir,
            target_frame=args.target_frame,
            allocentric_step=args.allocentric_step,
            allocentric_frames=allocentric_frames,
            external_camera_location=tuple(args.external_camera_location),
            external_camera_target=tuple(args.external_camera_target),
            blend_save_path=args.save_blend,
            hand_colors=hand_colors,
            object_color=object_color,
            render_camera=args.render_camera,
            render_trail=args.render_trail,
            render_allocentric=args.render_allocentric,
            render_target_frame=args.render_target_frame,
            render_hand=args.render_hand,
            render_obj_trail=args.render_obj_trail,
            vis_contact=args.vis_contact,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

