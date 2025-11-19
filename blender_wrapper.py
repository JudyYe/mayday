"""
001882_000030
High-level wrapper that creates Blender asset bundles and renders them.

Output tree:
    outputs/blender_results/<method>/<seq_obj>/
        blender_bundle/{t:04d}.pkl
        images/*.png
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio
import numpy as np
from fire import Fire
from jutils import web_utils
from PIL import Image
from tqdm import tqdm

from mayday.blender_cvt import (
    compute_allocentric_camera_from_assets,
    convert_npz_to_asset_list,
    convert_pkl_to_asset_list,
)

sample_list = [
    ["001882_000030_0_sample_0000000", "001882_000030_3_sample_0000000"],
    ["001911_000017_2_sample_0000000", "001911_000017_4_sample_0000000"],
    ["001974_000005_3_sample_0000000", "001974_000005_0_sample_0000000"],
    ["002007_000011_2_sample_0000000", "002007_000011_3_sample_0000000"],
    ["002031_000012_4_sample_0000000", "002031_000012_2_sample_0000000"],
    ["003150_000006_2_sample_0000000", "003150_000006_0_sample_0000000"],
]



LOGGER = logging.getLogger("mayday.blender_wrapper")

SAVE_ROOT = Path("outputs/blender_results")

METHOD_TO_SOURCE = {
    "teaser": "outputs/blender_results/teaser/{seq_obj}.pkl",
    "ours-gen": "outputs/org/ours/sample/{seq_obj}.pkl",
    "gt": "outputs/org/gt/{seq_obj}.pkl",
    "ours": "outputs/org/ours/post/{seq_obj}.pkl",
    "fp_simple": "outputs/org/ab_simple/post/{seq_obj}.pkl",
    "fp_full": "outputs/org/ab_full/post/{seq_obj}.pkl",
    "fp": "outputs/org/fp/{seq_obj}.npz",
}

METHOD_TO_COLOR = {
    "ours-gen": "blue1",
    "gt": "green",
    "fp": "yellow",
    "fp_simple": "purple",
    "fp_full": "pink",
    "ours": "blue1",
}
for i in range(5):
    METHOD_TO_SOURCE["ours-gen-%d" % i] = "outputs/org/ours-gen/sample_%d/{seq_obj}.pkl" % i
    METHOD_TO_COLOR["ours-gen-%d" % i] = "blue1"

DEFAULT_HAND_COLORS = "blue1,blue2"
# DEFAULT_TARGET_FRAME = 50
DEFAULT_TARGET_FRAME_LIST = [0, 50, 100, ]
DEFAULT_ALLOCENTRIC_STEP = 50
DEFAULT_BLENDER_EXE = "/move/u/yufeiy2/Package/blender-4.3.2-linux-x64/blender"
WEB_ROOT = SAVE_ROOT / "web"


def _ensure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _resolve_prediction_path(method: str, seq_obj: str) -> Path:
    if method not in METHOD_TO_SOURCE:
        raise KeyError(
            f"Unknown method '{method}'. Supported methods: {sorted(METHOD_TO_SOURCE)}"
        )
    template = METHOD_TO_SOURCE[method]
    candidate = Path(template.format(seq_obj=seq_obj)).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Prediction bundle not found for method '{method}': {candidate}"
        )
    return candidate


def _prepare_bundle(prediction_path: Path, bundle_dir: Path) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    LOGGER.info("Converting %s into bundles under %s", prediction_path, bundle_dir)
    if prediction_path.suffix == ".pkl":
        convert_pkl_to_asset_list(pkl_dir=str(prediction_path), output_dir=str(bundle_dir))
    elif prediction_path.suffix == ".npz":
        convert_npz_to_asset_list(npz_dir=str(prediction_path), output_dir=str(bundle_dir))
    else:
        raise ValueError(f"Unknown file extension: {prediction_path}. Must be .pkl or .npz")


def _run_blender(
    blender_exe: Path,
    bundle_dir: Path,
    image_dir: Path,
    *,
    object_color: str,
    hand_colors: str = DEFAULT_HAND_COLORS,
    target_frame: int = 0,
    allocentric_step: int = DEFAULT_ALLOCENTRIC_STEP,
    allocentric_frames: Sequence[int] | None = None,
    render_allocentric: bool = True,
    render_target_frame: bool = True,
    render_camera: bool = False,
    render_hand: bool = True,
    render_obj_trail: bool = False,
    render_trail: bool = True,
    vis_contact: bool = False,
    save_blend_path: Path | None = None,
    render_video: bool = False,
    video_frame_idx: int | None = None,
    render_width: int = 1440,
    render_height: int = 1080,
    render_cam_h: int | None = None,
    render_samples: int = 64,
    dynamic_floor: bool = False,
) -> None:
    cmd = [
        str(blender_exe),
        "--background",
        "--python",
        "mayday/blender_vis.py",
        "--",
        "--mode",
        "render_assets",
        "--asset-dir",
        str(bundle_dir),
        "--output-dir",
        str(image_dir),
        "--target-frame",
        str(target_frame),
        "--allocentric-step",
        str(allocentric_step),
        "--hand-color",
        hand_colors,
        "--object-color",
        object_color,
    ]
    if allocentric_frames is not None:
        frame_tokens = ",".join(str(int(idx)) for idx in allocentric_frames)
        cmd.extend(["--allocentric-frames", frame_tokens])
    if render_camera:
        cmd.append("--render-camera")
    cmd.append("--render-allocentric" if render_allocentric else "--no-render-allocentric")
    cmd.append("--render-target-frame" if render_target_frame else "--no-render-target-frame")
    cmd.append("--render-trail" if render_trail else "--no-render-trail")
    cmd.append("--vis-contact" if vis_contact else "--no-vis-contact")
    cmd.append("--render-hand" if render_hand else "--no-render-hand")
    cmd.append("--vis-obj-trail" if render_obj_trail else "--no-vis-obj-trail")
    if save_blend_path is not None:
        cmd.extend(["--save-blend", str(save_blend_path)])
    if render_video:
        cmd.append("--render-video")
        if video_frame_idx is not None:
            cmd.extend(["--video-frame-idx", str(video_frame_idx)])
    cmd.extend(["--render-width", str(render_width)])
    cmd.extend(["--render-height", str(render_height)])
    if render_cam_h is not None:
        cmd.extend(["--render-cam-h", str(render_cam_h)])
    cmd.extend(["--render-samples", str(render_samples)])
    if dynamic_floor:
        cmd.append("--dynamic-floor")
    print("Running Blender render: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _gather_scene_vertices(bundle_dir: Path) -> Tuple[List[np.ndarray], List[Path]]:
    asset_paths = sorted(bundle_dir.glob("*.pkl"))
    scene_vertices: List[np.ndarray] = []
    for asset_path in asset_paths:
        with open(asset_path, "rb") as f:
            bundle = pickle.load(f)
        for mesh in bundle.get("meshes", []):
            verts = mesh.get("vertices")
            if verts is None:
                continue
            verts_np = np.asarray(verts, dtype=np.float32)
            if verts_np.ndim == 2 and verts_np.shape[1] == 3:
                scene_vertices.append(verts_np)
        camera = bundle.get("camera")
        if camera is not None:
            extr = camera.get("extrinsic_wTc")
            if extr is not None:
                extr_np = np.asarray(extr, dtype=np.float32)
                if extr_np.shape == (4, 4):
                    scene_vertices.append(extr_np[:3, 3][None, :])
    return scene_vertices, asset_paths


def _rename_outputs(image_dir: Path, seq_obj: str, method: str, web_dir: Path, no_hand: bool = False) -> None:
    web_dir.mkdir(parents=True, exist_ok=True)
    for path in list(image_dir.glob("*.png")):
        stem = path.stem
        suffix = path.suffix
        suffix_str = "_no_hand" if no_hand else ""
        new_name = f"{seq_obj}_{stem}_{method}{suffix_str}{suffix}"
        target_path = web_dir / new_name
        if target_path.exists():
            continue
        shutil.copy2(path, target_path)


def _overlay_camera_on_input(camera_path: Path, input_path: Path, output_path: Path, alpha: float = 0.7) -> None:
    """Overlay camera render on top of input image with proper alpha handling."""
    if not camera_path.exists() or not input_path.exists():
        return
    
    import numpy as np
    from PIL import ImageFilter
    
    # Load images
    camera_img = Image.open(camera_path)
    input_img = Image.open(input_path).convert("RGB")
    
    # Convert camera to RGBA if needed
    if camera_img.mode != "RGBA":
        camera_img = camera_img.convert("RGBA")
    
    # Resize camera image to match input if needed
    if camera_img.size != input_img.size:
        camera_img = camera_img.resize(input_img.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays for better alpha handling
    camera_arr = np.array(camera_img, dtype=np.float32) / 255.0
    input_arr = np.array(input_img, dtype=np.float32) / 255.0
    
    # Extract alpha channel from camera image
    camera_rgb = camera_arr[:, :, :3]
    camera_alpha = camera_arr[:, :, 3:4]  # Keep as (H, W, 1) for broadcasting
    
    # Create a mask: use camera alpha if available, otherwise use a threshold on brightness
    # to avoid artifacts at boundaries
    if np.any(camera_alpha > 0.01):  # If there's actual alpha data
        mask = camera_alpha
    else:
        # Fallback: create mask from non-black pixels (assuming black background)
        brightness = np.mean(camera_rgb, axis=2, keepdims=True)
        mask = (brightness > 0.01).astype(np.float32)
    
    # Apply smoothing to mask edges to reduce artifacts using PIL's Gaussian blur
    # Convert mask to PIL Image, blur it, then convert back
    mask_2d = (mask[:, :, 0] * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_2d, mode='L')
    mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask = np.array(mask_blurred, dtype=np.float32)[:, :, None] / 255.0
    
    # Blend: input * (1 - mask * alpha) + camera * (mask * alpha)
    overlay_alpha = mask * alpha
    blended = input_arr * (1 - overlay_alpha) + camera_rgb * overlay_alpha
    
    # Convert back to uint8 and save
    blended_uint8 = np.clip(blended * 255, 0, 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_uint8)
    blended_img.save(output_path)


def _create_video_from_frames(frame_dir: Path, output_video_path: Path, fps: int = 30, pattern: str = "*.png") -> None:
    """Create video from sequence of frames using imageio. Transparent areas are converted to white."""
    frame_paths = sorted(frame_dir.glob(pattern))
    if not frame_paths:
        LOGGER.warning(f"No frames found matching {pattern} in {frame_dir}")
        return
    
    frames = []
    for frame_path in frame_paths:
        try:
            img = imageio.imread(frame_path)
            # If image has alpha channel, composite with white background
            if img.shape[2] == 4:  # RGBA
                alpha = img[:, :, 3:4] / 255.0
                rgb = img[:, :, :3]
                # Composite: white * (1 - alpha) + rgb * alpha
                img = (255 * (1 - alpha) + rgb * alpha).astype(np.uint8)
            frames.append(img)
        except Exception as exc:
            LOGGER.warning(f"Failed to read frame {frame_path}: {exc}")
    
    if frames:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(output_video_path), frames, fps=fps, codec="libx264", quality=8)
        LOGGER.info(f"Created video: {output_video_path}")


def _index_web_images(image_folder: str, method_columns: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Dict[str, Path] | Path | None]]]:
    web_dir = WEB_ROOT / image_folder
    index: Dict[str, Dict[str, Dict[str, Dict[str, Path] | Path | None]]] = {}
    for png_path in web_dir.glob("*.png"):
        stem = png_path.stem
        for method in method_columns:
            suffix = f"_{method}"
            if not stem.endswith(suffix):
                continue
            base = stem[: -len(suffix)]
            if base.endswith("_"):
                base = base[:-1]
            tokens = base.split("_")
            if len(tokens) < 3:
                continue
            seq_obj = "_".join(tokens[:2])
            label_tokens = tokens[2:]
            method_entry = index.setdefault(seq_obj, {}).setdefault(
                method,
                {"overlay": None, "camera": {}, "input": {}},
            )
            if len(label_tokens) >= 2 and label_tokens[0] == "allocentric" and label_tokens[1] == "overlay":
                method_entry["overlay"] = png_path
            elif len(label_tokens) >= 2:
                frame = label_tokens[0]
                label = label_tokens[1]
                if label == "camera":
                    method_entry["camera"][frame] = png_path
                elif label == "input":
                    method_entry["input"][frame] = png_path
            break
    return index


def _build_webpage(image_folder: str, method_columns: Sequence[str]) -> None:
    web_dir = WEB_ROOT / image_folder
    web_dir.mkdir(parents=True, exist_ok=True)

    index = _index_web_images(image_folder, method_columns)
    seq_ids = sorted(index.keys())

    cell_list: List[List[str]] = []

    for seq_obj in seq_ids:
        cell_list.append([f"<b>{seq_obj}</b>"] + [""] * len(method_columns))

        method_data = {
            method: index.get(seq_obj, {}).get(method, {"overlay": None, "camera": {}, "input": {}})
            for method in method_columns
        }

        overlays_row: List[str] = ["allocentric_overlay"]
        for method in method_columns:
            overlay = method_data[method].get("overlay")
            overlays_row.append(str(overlay) if isinstance(overlay, Path) else f"{seq_obj}_allocentric_overlay_{method}.png")
        cell_list.append(overlays_row)

        frames = sorted(
            {
                frame
                for method in method_columns
                for frame in method_data[method].get("camera", {}).keys()
            }
        )

        for frame in frames:
            row: List[str] = []
            input_path = None
            for method in method_columns:
                candidate = method_data[method].get("input", {}).get(frame)
                if candidate is not None:
                    input_path = candidate
                    break
            row.append(str(input_path) if input_path else f"{seq_obj}_{frame}_input_gt.png")
            for method in method_columns:
                cam_path = method_data[method].get("camera", {}).get(frame)
                row.append(str(cam_path) if cam_path else f"{seq_obj}_{frame}_camera_{method}.png")
            cell_list.append(row)

    if cell_list:
        web_utils.run(
            html_root=str(web_dir / "vis.html"),
            cell_list=cell_list,
            width=200,
            inplace=True,
        )


def _compute_joint_allocentric_camera(
    method_info: List[Tuple[str, Path, Path, str, str, bool]],
    total_frames: int,
    divide: float = 1.0,
    dynamic_floor: bool = False,
) -> Dict:
    """Compute allocentric camera from all frames across all methods."""
    all_vertices: List[np.ndarray] = []
    
    for method_idx, (method, bundle_dir, _, _, _, render_hand_flag) in enumerate(method_info):
        # Get asset paths
        asset_paths = sorted(bundle_dir.glob("*.pkl"))
        if not asset_paths:
            continue
        
        # Compute offset (spacing methods along X axis)
        offset = np.array([method_idx * divide, 0.0, 0.0], dtype=np.float32)
        
        # Collect vertices from all frames
        for frame_idx in range(min(total_frames, len(asset_paths))):
            with open(asset_paths[frame_idx], "rb") as f:
                frame_data = pickle.load(f)
            
            # Process meshes
            for mesh_info in frame_data.get("meshes", []):
                name_lower = mesh_info.get("name", "").lower()
                
                # Skip hands if render_hand_flag is False
                if not render_hand_flag and "hand" in name_lower:
                    continue
                
                verts = mesh_info.get("vertices")
                if verts is None or len(verts) == 0:
                    continue
                
                # Translate vertices and add to collection
                translated_vertices = np.asarray(verts, dtype=np.float32) + offset[np.newaxis, :]
                all_vertices.append(translated_vertices)
    
    if not all_vertices:
        raise ValueError("No geometry available to compute allocentric camera")
    
    # Compute allocentric camera from all vertices across all frames
    alloc_camera = compute_allocentric_camera_from_assets(all_vertices)
    
    # Compute floor_z based on dynamic_floor flag
    if dynamic_floor:
        # Compute minimum Z coordinate (lowest vertex) across all vertices
        all_vertices_array = np.concatenate(all_vertices, axis=0)
        min_z = float(np.min(all_vertices_array[:, 2]))
        floor_z = min_z - 0.05
    else:
        # Default floor position
        floor_z = -1.0
    
    result = {
        "intrinsic": alloc_camera["intrinsic"],
        "extrinsic_wTc": alloc_camera["extrinsic_wTc"],
        "center": alloc_camera["center"].tolist(),
        "radius": float(alloc_camera["radius"]),
        "width": int(alloc_camera["width"]),
        "height": int(alloc_camera["height"]),
    }
    
    # Only include floor_z if dynamic_floor is enabled
    if dynamic_floor:
        result["floor_z"] = floor_z
    
    return result


def _build_joint_bundle_for_frame(
    method_info: List[Tuple[str, Path, Path, str, str, bool]],
    frame_idx: int,
    divide: float = 1.0,
    vis_contact: bool = False,
    alloc_camera: Dict | None = None,
) -> Dict:
    """Build a joint bundle combining all methods with offsets for a single frame."""
    meshes: List[Dict] = []
    
    from mayday.blender_vis import COLOR_CHOICES
    
    for method_idx, (method, bundle_dir, _, object_color_name, hand_color_name, render_hand_flag) in enumerate(method_info):
        # Get asset paths
        asset_paths = sorted(bundle_dir.glob("*.pkl"))
        if not asset_paths or frame_idx >= len(asset_paths):
            continue
        
        # Load frame data
        with open(asset_paths[frame_idx], "rb") as f:
            frame_data = pickle.load(f)
        
        # Compute offset (spacing methods along X axis)
        offset = np.array([method_idx * divide, 0.0, 0.0], dtype=np.float32)
        
        # Get method's color from METHOD_TO_COLOR - use same color for both objects and hands
        method_color_name = METHOD_TO_COLOR.get(method, "blue1")
        method_color_rgb = COLOR_CHOICES.get(method_color_name, COLOR_CHOICES["blue1"])
        
        # Process meshes
        for mesh_info in frame_data.get("meshes", []):
            name_lower = mesh_info.get("name", "").lower()
            
            # Skip hands if render_hand_flag is False
            if not render_hand_flag and "hand" in name_lower:
                continue
            
            verts = mesh_info.get("vertices")
            if verts is None or len(verts) == 0:
                continue
            
            # Translate vertices
            translated_vertices = np.asarray(verts, dtype=np.float32) + offset[np.newaxis, :]
            
            # Create mesh copy with offset
            mesh_copy = dict(mesh_info)
            mesh_copy["vertices"] = translated_vertices.tolist()
            mesh_copy["name"] = f"{method}_{mesh_info['name']}"
            
            # Apply method's color to both objects and hands
            original_color = mesh_info.get("color", [0.8, 0.8, 0.8, 1.0])
            
            # Handle contact visualization first
            if vis_contact and len(original_color) >= 3:
                color_arr = np.asarray(original_color[:3], dtype=np.float32)
                if np.allclose(color_arr, 1.0, atol=1e-3):
                    mesh_copy["color"] = [*COLOR_CHOICES["red"], original_color[3] if len(original_color) > 3 else 1.0]
                    mesh_copy["force_color"] = True
                else:
                    # Use method color and force it
                    mesh_copy["color"] = [*method_color_rgb, original_color[3] if len(original_color) > 3 else 1.0]
                    mesh_copy["force_color"] = True
            else:
                # Use method color and force it
                mesh_copy["color"] = [*method_color_rgb, original_color[3] if len(original_color) > 3 else 1.0]
                mesh_copy["force_color"] = True
            meshes.append(mesh_copy)
    
    if not meshes:
        raise ValueError("No geometry available to build joint bundle")
    
    # Use provided allocentric camera or compute from current frame (fallback)
    if alloc_camera is None:
        # Fallback: compute from current frame only (shouldn't happen in normal flow)
        aggregated_vertices = []
        for mesh in meshes:
            verts = mesh.get("vertices")
            if verts:
                aggregated_vertices.append(np.asarray(verts, dtype=np.float32))
        alloc_camera = compute_allocentric_camera_from_assets(aggregated_vertices)
        alloc_camera = {
            "intrinsic": alloc_camera["intrinsic"],
            "extrinsic_wTc": alloc_camera["extrinsic_wTc"],
            "center": alloc_camera["center"].tolist(),
            "radius": float(alloc_camera["radius"]),
            "width": int(alloc_camera["width"]),
            "height": int(alloc_camera["height"]),
        }
    
    return {
        "meshes": meshes,
        "camera": None,  # No camera view in joint mode
        "alloc_camera": alloc_camera,
        "image_path": None,
    }


def render_all_methods(
    seq_obj: str,
    method_list: Sequence[str],
    *,
    image_folder: str = "images",
    cvt_bundle: bool = True,
    render: bool = True,
    target_frames: Sequence[int] | None = None,
    allocentric_step: int = DEFAULT_ALLOCENTRIC_STEP,
    blender_exe: str | None = None,
    align_alloc: bool = False,
    vis_obj_trail: bool = False,
    render_hand_per_method: Sequence[int] | str | None = None,
    render_video: bool = False,
    render_video_alloc_joint: bool = False,
    video_fps: int = 30,
    divide: float = 1.0,
    render_width: int = 1440,
    render_height: int = 1080,
    render_cam_h: int | None = None,
    render_samples: int = 64,
    dynamic_floor: bool = False,
    no_hand: bool = False,
    **kwargs,
) -> None:
    """Convert and render the specified methods for a sequence/object pair."""
    _ensure_logger()

    if not seq_obj:
        raise ValueError("seq_obj must be a non-empty string.")

    blender_exec = Path(
        blender_exe or os.environ.get("BLENDER_EXE", DEFAULT_BLENDER_EXE)
    ).resolve()
    if render and not blender_exec.is_file():
        raise FileNotFoundError(
            f"Blender executable not found at {blender_exec}. "
            "Set BLENDER_EXE or pass blender_exe explicitly."
        )

    method_info: List[Tuple[str, Path, Path, str, str, bool]] = []

    # If --no_hand is set, override render_hand
    LOGGER.info(f"render_all_methods: no_hand={no_hand}, kwargs.render_hand={kwargs.get('render_hand', 'not set')}")
    if no_hand:
        base_render_hand = False
    else:
        base_render_hand = bool(kwargs.get("render_hand", True))
    LOGGER.info(f"render_all_methods: base_render_hand={base_render_hand}")
    if render_hand_per_method is None:
        method_hand_flags = [base_render_hand for _ in method_list]
    else:
        if isinstance(render_hand_per_method, str):
            tokens = [token.strip() for token in render_hand_per_method.split(",") if token.strip()]
            values = tokens
        else:
            values = list(render_hand_per_method)
        if len(values) != len(method_list):
            raise ValueError(
                f"render_hand_per_method expects {len(method_list)} entries, got {len(values)}"
            )
        method_hand_flags = []
        for value in values:
            if isinstance(value, str):
                if value not in {"0", "1"}:
                    raise ValueError("render_hand_per_method entries must be '0' or '1'")
                method_hand_flags.append(value == "1")
            else:
                method_hand_flags.append(bool(value))

    for idx, method in enumerate(method_list):
        prediction_path = _resolve_prediction_path(method, seq_obj)
        color_entry = METHOD_TO_COLOR.get(method, "blue1")
        if isinstance(color_entry, tuple):
            hand_color_entry, object_color = color_entry
        else:
            hand_color_entry, object_color = color_entry, color_entry

        method_root = SAVE_ROOT / method / seq_obj
        bundle_dir = method_root / "blender_bundle"
        image_dir = method_root / image_folder

        if cvt_bundle:
            _prepare_bundle(prediction_path, bundle_dir)
        elif not bundle_dir.exists():
            raise FileNotFoundError(
                f"Bundle directory {bundle_dir} is missing. "
                "Run with cvt_bundle=True first."
            )

        method_info.append((method, bundle_dir, image_dir, object_color, hand_color_entry, method_hand_flags[idx]))

    shared_alloc = None
    if align_alloc:
        combined_vertices: List[np.ndarray] = []
        assets_map: Dict[Path, List[Path]] = {}
        for _, bundle_dir, _, _, _, _ in method_info:
            scene_vertices, asset_paths = _gather_scene_vertices(bundle_dir)
            combined_vertices.extend(scene_vertices)
            assets_map[bundle_dir] = asset_paths
        if not combined_vertices:
            raise ValueError("Unable to compute allocentric camera: no scene geometry found.")
        shared_alloc = compute_allocentric_camera_from_assets(combined_vertices)
        shared_alloc["center"] = shared_alloc["center"].tolist()
        for asset_paths in assets_map.values():
            for asset_path in asset_paths:
                with open(asset_path, "rb") as f:
                    bundle = pickle.load(f)
                bundle["alloc_camera"] = shared_alloc
                with open(asset_path, "wb") as f:
                    pickle.dump(bundle, f)

    frame_list = list(target_frames) if target_frames is not None else list(DEFAULT_TARGET_FRAME_LIST)
    # if not frame_list:
    #     frame_list = [0]

    if render:
        web_dir = WEB_ROOT / image_folder
        web_dir.mkdir(parents=True, exist_ok=True)
        
        if render_video_alloc_joint:
            # Joint allocentric video rendering mode: all methods in one scene
            LOGGER.info(f"Rendering joint allocentric video for {seq_obj} with methods: {method_list}")
            
            # Determine total frames from first method (assuming all have same length)
            if not method_info:
                raise ValueError("No methods to render")
            
            first_bundle_dir = method_info[0][1]
            asset_paths = sorted(first_bundle_dir.glob("*.pkl"))
            if not asset_paths:
                raise ValueError(f"No asset bundles found in {first_bundle_dir}")
            
            total_frames = len(asset_paths)
            LOGGER.info(f"Rendering joint video: {total_frames} frames")
            
            # Compute allocentric camera once from all frames across all methods
            LOGGER.info("Computing allocentric camera from all frames...")
            shared_alloc_camera = _compute_joint_allocentric_camera(
                method_info,
                total_frames,
                divide=divide,
                dynamic_floor=dynamic_floor,
            )
            LOGGER.info("Allocentric camera computed")
            
            # Create method name for folder: cmp_{method1+method2+...}
            method_folder_name = f"cmp_{'+'.join(method_list)}"
            
            # Create output directory for joint video
            joint_output_dir = SAVE_ROOT / method_folder_name / seq_obj / image_folder
            joint_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary bundle directory for joint frames
            joint_bundle_dir = SAVE_ROOT / method_folder_name / seq_obj / "blender_bundle"
            joint_bundle_dir.mkdir(parents=True, exist_ok=True)
            
            # Create video frames directory
            video_alloc_dir = joint_output_dir / "video_alloc_frames"
            video_alloc_dir.mkdir(exist_ok=True)
            
            # Build joint bundles for each frame and render
            for frame_idx in tqdm(range(total_frames), desc="Rendering joint video"):
                # Build joint bundle for this frame (using shared allocentric camera)
                joint_bundle = _build_joint_bundle_for_frame(
                    method_info,
                    frame_idx,
                    divide=divide,
                    vis_contact=kwargs.get("vis_contact", False),
                    alloc_camera=shared_alloc_camera,
                )
                
                # Save joint bundle
                joint_bundle_path = joint_bundle_dir / f"{frame_idx:04d}.pkl"
                with open(joint_bundle_path, "wb") as f:
                    pickle.dump(joint_bundle, f)
                
                # Render allocentric view only (no camera view, no trails, only current frame)
                _run_blender(
                    blender_exec,
                    joint_bundle_dir,
                    video_alloc_dir,
                    object_color="pink",  # Not used in joint mode (colors set in bundle)
                    hand_colors=DEFAULT_HAND_COLORS,  # Not used in joint mode (colors set in bundle)
                    target_frame=frame_idx,
                    allocentric_step=1,
                    allocentric_frames=[frame_idx],  # Only current frame
                    render_allocentric=True,
                    render_target_frame=False,  # No camera view
                    render_camera=False,  # No camera wireframes
                    render_hand=True,
                    render_trail=False,  # No hand/camera trails
                    vis_contact=kwargs.get("vis_contact", False),
                    render_video=True,
                    video_frame_idx=frame_idx,
                    render_width=render_width,
                    render_height=render_height,
                    render_samples=render_samples,
                    dynamic_floor=dynamic_floor,
                )
            
            # Create video
            suffix_str = "_no_hand" if no_hand else ""
            LOGGER.info(f"Creating joint allocentric video with no_hand={no_hand}, suffix='{suffix_str}'")
            alloc_video_path = joint_output_dir / f"{seq_obj}_joint_alloc{suffix_str}.mp4"
            LOGGER.info(f"Video path: {alloc_video_path}")
            _create_video_from_frames(video_alloc_dir, alloc_video_path, fps=video_fps, pattern="*_allocentric_overlay.png")
            
            # Copy video to web directory
            if alloc_video_path.exists():
                web_video_path = web_dir / alloc_video_path.name
                shutil.copy2(alloc_video_path, web_video_path)
                LOGGER.info(f"Created joint allocentric video: {alloc_video_path}")
        
        elif render_video:
            # Video rendering mode: render all frames
            for method, bundle_dir, image_dir, object_color, hand_color_entry, render_hand_flag in method_info:
                image_dir.mkdir(parents=True, exist_ok=True)
                
                # Get total number of frames from bundle directory
                asset_paths = sorted(bundle_dir.glob("*.pkl"))
                if not asset_paths:
                    LOGGER.warning(f"No asset bundles found in {bundle_dir}")
                    continue
                
                total_frames = len(asset_paths)
                LOGGER.info(f"Rendering video for {method}: {total_frames} frames")
                
                # Create temporary directories for video frames
                video_camera_dir = image_dir / "video_camera_frames"
                video_alloc_dir = image_dir / "video_alloc_frames"
                video_overlay_dir = image_dir / "video_overlay_frames"
                video_camera_dir.mkdir(exist_ok=True)
                video_alloc_dir.mkdir(exist_ok=True)
                video_overlay_dir.mkdir(exist_ok=True)
                
                # Render each frame
                for frame_idx in tqdm(range(total_frames), desc=f"Rendering {method} video"):
                    # Render camera view for this frame (square, uses render_cam_h)
                    _run_blender(
                        blender_exec,
                        bundle_dir,
                        video_camera_dir,
                        object_color=object_color,
                        hand_colors=hand_color_entry,
                        target_frame=frame_idx,
                        allocentric_step=allocentric_step,
                        render_allocentric=False,
                        render_target_frame=True,
                        render_camera=kwargs.get("render_camera", False),
                        render_hand=render_hand_flag,
                        render_obj_trail=vis_obj_trail,
                        vis_contact=kwargs.get("vis_contact", False),
                        render_video=True,
                        video_frame_idx=frame_idx,
                        render_width=render_width,
                        render_height=render_height,
                        render_cam_h=render_cam_h,
                        render_samples=render_samples,
                    )
                    
                    # Render allocentric view for current frame only
                    _run_blender(
                        blender_exec,
                        bundle_dir,
                        video_alloc_dir,
                        object_color=object_color,
                        hand_colors=hand_color_entry,
                        target_frame=frame_idx,
                        allocentric_step=1,
                        allocentric_frames=[frame_idx],
                        render_allocentric=True,
                        render_target_frame=False,
                        render_camera=kwargs.get("render_camera", False),
                        render_hand=render_hand_flag,
                        render_obj_trail=vis_obj_trail,
                        vis_contact=kwargs.get("vis_contact", False),
                        render_video=True,
                        video_frame_idx=frame_idx,
                        render_width=render_width,
                        render_height=render_height,
                        render_samples=render_samples,
                    )
                    
                    # Overlay camera view on input image
                    camera_frame_path = video_camera_dir / f"{frame_idx:04d}_camera.png"
                    input_frame_path = video_camera_dir / f"{frame_idx:04d}_input.png"
                    overlay_frame_path = video_overlay_dir / f"{frame_idx:04d}_overlay.png"
                    
                    if camera_frame_path.exists() and input_frame_path.exists():
                        _overlay_camera_on_input(camera_frame_path, input_frame_path, overlay_frame_path)
                
                # Create videos
                suffix_str = "_no_hand" if no_hand else ""
                LOGGER.info(f"Creating videos for {method} with no_hand={no_hand}, suffix='{suffix_str}'")
                camera_video_path = image_dir / f"{seq_obj}_{method}_camera{suffix_str}.mp4"
                alloc_video_path = image_dir / f"{seq_obj}_{method}_alloc{suffix_str}.mp4"
                overlay_video_path = image_dir / f"{seq_obj}_{method}_overlay{suffix_str}.mp4"
                LOGGER.info(f"Video paths: camera={camera_video_path}, alloc={alloc_video_path}, overlay={overlay_video_path}")
                
                _create_video_from_frames(video_camera_dir, camera_video_path, fps=video_fps, pattern="*_camera.png")
                _create_video_from_frames(video_alloc_dir, alloc_video_path, fps=video_fps, pattern="*_allocentric_overlay.png")
                _create_video_from_frames(video_overlay_dir, overlay_video_path, fps=video_fps, pattern="*_overlay.png")
                
                # Copy videos to web directory
                for video_path in [camera_video_path, alloc_video_path, overlay_video_path]:
                    if video_path.exists():
                        web_video_path = web_dir / video_path.name
                        shutil.copy2(video_path, web_video_path)
        else:
            # Original image rendering mode
            for method, bundle_dir, image_dir, object_color, hand_color_entry, render_hand_flag in method_info:
                image_dir.mkdir(parents=True, exist_ok=True)

                # overlay_frame = frame_list[0]
                _run_blender(
                    blender_exec,
                    bundle_dir,
                    image_dir,
                    object_color=object_color,
                    hand_colors=hand_color_entry,
                    target_frame=0,
                    allocentric_step=allocentric_step,
                    render_allocentric=True,
                    render_target_frame=False,
                    render_camera=kwargs.get("render_camera", False),
                    render_hand=render_hand_flag,
                    render_obj_trail=vis_obj_trail,
                    vis_contact=kwargs.get("vis_contact", False),
                )
                _rename_outputs(image_dir, seq_obj, method, web_dir, no_hand=no_hand)

                for tf in frame_list:
                    _run_blender(
                        blender_exec,
                        bundle_dir,
                        image_dir,
                        object_color=object_color,
                        hand_colors=hand_color_entry,
                        target_frame=tf,
                        allocentric_step=allocentric_step,
                        render_allocentric=False,
                        render_target_frame=True,
                        render_camera=kwargs.get("render_camera", False),
                        render_hand=render_hand_flag,
                        render_obj_trail=vis_obj_trail,
                        vis_contact=kwargs.get("vis_contact", False),
                    )
                    _rename_outputs(image_dir, seq_obj, method, web_dir, no_hand=no_hand)


def rebuild_web_gallery(image_folder: str = "images", method_columns: Sequence[str] = ("gt", "fp_simple", "fp_full", "ours")) -> None:
    _build_webpage(image_folder, method_columns)


def render_samples_from_list(
    sample_list: Sequence[Sequence[str]],
    *,
    blender_exe: str | None = None,
    video_fps: int = 30,
    render_width: int = 1440,
    render_height: int = 1080,
    render_samples: int = 64,
    dynamic_floor: bool = False,
    **kwargs,
) -> None:
    """Render videos from sample_list.
    
    Decode rule: {seq_obj}_{s}_sample_00000000
    - seq_obj: sequence/object identifier (e.g., "001882_000030")
    - s: sample variant number (e.g., "0")
    - i: sample index from sample_00000000 (e.g., 0)
    
    PKL file location: outputs/org/ours-gen/sample_%d/{seq_obj}.pkl where %d is i
    
    Output: blender_results/sample/{seq_obj}_{i}_sample.mp4
    """
    _ensure_logger()
    
    if blender_exe is None:
        blender_exe = Path(DEFAULT_BLENDER_EXE)
    else:
        blender_exe = Path(blender_exe)
    
    if not blender_exe.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_exe}")
    
    for sample_group in tqdm(sample_list, desc="Processing sample groups"):
        for entry_idx, sample_entry in enumerate(sample_group):
            # Parse: {seq_obj}_{s}_sample_00000000
            # Example: "001882_000030_0_sample_0000000"
            parts = sample_entry.split("_sample_")
            if len(parts) != 2:
                LOGGER.warning(f"Invalid sample entry format: {sample_entry}, skipping")
                continue
            
            prefix = parts[0]  # "001882_000030_0"
            
            # Extract seq_obj and s from prefix
            # prefix is like "001882_000030_0", we need to find where the sample variant starts
            # The last number before "sample" is s, everything before that is seq_obj
            prefix_parts = prefix.split("_")
            if len(prefix_parts) < 3:
                LOGGER.warning(f"Invalid prefix format: {prefix}, skipping")
                continue
            
            # The last part is s, everything else is seq_obj
            s = prefix_parts[-1]
            seq_obj = "_".join(prefix_parts[:-1])
            
            # Use entry index as the video index (0 for first entry, 1 for second, etc.)
            # This ensures each entry in a group produces a different output file
            i = entry_idx
            
            # Find pkl file: outputs/org/ours-gen/sample_%d/{seq_obj}.pkl
            # Use entry_idx to determine which sample PKL file to use
            pkl_path = Path(f"outputs/org/ours-gen/sample_{i}/{seq_obj}.pkl")
            if not pkl_path.exists():
                LOGGER.warning(f"PKL file not found: {pkl_path}, skipping")
                continue
            
            # Check if video already exists - skip if it does
            output_video_path = SAVE_ROOT / "sample" / f"{seq_obj}_{i}_sample.mp4"
            if output_video_path.exists():
                LOGGER.info(f"Video already exists, skipping: {output_video_path}")
                continue
            
            LOGGER.info(f"Processing sample: {sample_entry} -> seq_obj={seq_obj}, s={s}, i={i}")
            
            # Prepare bundle directory
            bundle_dir = SAVE_ROOT / "sample" / f"{seq_obj}_{i}_sample" / "blender_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert pkl to bundle
            _prepare_bundle(pkl_path, bundle_dir)
            
            # Get number of frames
            asset_paths = sorted(bundle_dir.glob("*.pkl"))
            num_frames = len(asset_paths)
            if num_frames == 0:
                LOGGER.warning(f"No frames found in bundle: {bundle_dir}, skipping")
                continue
            
            # Render allocentric view for all frames
            output_dir = SAVE_ROOT / "sample" / f"{seq_obj}_{i}_sample" / "images"
            video_frames_dir = output_dir / "video_alloc_frames"
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            
            LOGGER.info(f"Rendering {num_frames} frames for {seq_obj}_{i}_sample")
            
            # Render each frame
            for frame_idx in tqdm(range(num_frames), desc=f"Rendering {seq_obj}_{i}_sample"):
                _run_blender(
                    blender_exe,
                    bundle_dir,
                    video_frames_dir,
                    object_color="pink",  # Not critical for allocentric view
                    hand_colors=DEFAULT_HAND_COLORS,
                    target_frame=frame_idx,
                    allocentric_step=1,
                    allocentric_frames=[frame_idx],  # Only current frame
                    render_allocentric=True,
                    render_target_frame=False,  # No camera view
                    render_camera=False,  # No camera wireframes
                    render_hand=True,
                    render_trail=False,  # No trails
                    vis_contact=True,  # Use vis_contact=True as specified
                    render_video=True,
                    video_frame_idx=frame_idx,
                    render_width=render_width,
                    render_height=render_height,
                    render_samples=render_samples,
                    dynamic_floor=dynamic_floor,
                )
            
            # Create video
            LOGGER.info(f"Creating video: {output_video_path}")
            _create_video_from_frames(
                video_frames_dir,
                output_video_path,
                fps=video_fps,
                pattern="*_allocentric_overlay.png"
            )
            
            if output_video_path.exists():
                LOGGER.info(f"Created sample video: {output_video_path}")
            else:
                LOGGER.warning(f"Failed to create sample video: {output_video_path}")


def main(
    seq_obj: str | None = None,
    method_list: Sequence[str] = ("gt", "fp_simple", "fp_full", "ours"),
    *,
    mode: str = "render",
    target_frames: Sequence[int] | None = None,
    dry: bool = False,
    render_video: bool = False,
    render_video_alloc_joint: bool = False,
    video_fps: int = 30,
    divide: float = 1.0,
    render_width: int = 1440,
    render_height: int = 1080,
    render_cam_h: int | None = None,
    render_samples: int = 64,
    dynamic_floor: bool = False,
    no_hand: bool = False,
    skip=False,
    **kwargs,
) -> None:
    image_folder = kwargs.get("image_folder", "images")
    kwargs["image_folder"] = image_folder

    if mode == "render":
    # if not dry:
        if seq_obj is None:
            split = kwargs.get("split", "test50obj")
            split_file = osp.join("data/HOT3D-CLIP", "sets", "split.json")
            with open(split_file, "r", encoding="utf-8") as f:
                split_dict = json.load(f)
            seq_obj_list = split_dict[split]
        else:
            seq_obj_list = [seq_obj]

        for s, seq in enumerate(tqdm(seq_obj_list)):
            lock_file = SAVE_ROOT / f"lock.{image_folder}_{'+'.join(method_list)}_joint{render_video_alloc_joint}" / f"{seq}"
            done_file = SAVE_ROOT / f"done.{image_folder}_{'+'.join(method_list)}_joint{render_video_alloc_joint}" / f"{seq}"
            if skip and osp.exists(done_file):
                continue
            try:
                os.makedirs(lock_file)
            except FileExistsError:
                if skip:
                    continue

            render_all_methods(
                seq,
                method_list,
                target_frames=target_frames,
                render_video=render_video,
                render_video_alloc_joint=render_video_alloc_joint,
                video_fps=video_fps,
                divide=divide,
                render_width=render_width,
                render_height=render_height,
                render_cam_h=render_cam_h,
                render_samples=render_samples,
                dynamic_floor=dynamic_floor,
                no_hand=no_hand,
                **kwargs,
            )
            os.makedirs(done_file)
            os.rmdir(lock_file)

    elif mode == "web":
        rebuild_web_gallery(image_folder, method_list)
    elif mode == "render_sample":
        # Render from sample_list
        render_samples_from_list(
            sample_list,
            blender_exe=kwargs.get("blender_exe", DEFAULT_BLENDER_EXE),
            video_fps=video_fps,
            render_width=render_width,
            render_height=render_height,
            render_samples=render_samples,
            dynamic_floor=dynamic_floor,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")



if __name__ == "__main__":
    Fire(main)
