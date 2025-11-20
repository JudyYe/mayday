from __future__ import annotations

import os
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from fire import Fire
from tqdm import tqdm

from mayday.blender_cvt import compute_allocentric_camera_from_assets, convert_pkl_to_asset_list
from mayday.blender_wrapper import _run_blender, _create_video_from_frames, _overlay_camera_on_input  # reuse existing helper
from mayday.blender_vis  import COLOR_CHOICES

DEFAULT_SPACING = 0.0
DEFAULT_COLORS = [ "green", "purple", "yellow", "pink", "lightgreen"]
DEFAULT_TARGET_FRAMES = [0, 50, 100]
DEFAULT_HAND_COLORS = "blue1,blue2"

# Object colors for video teaser (max 10 objects)
OBJECT_COLORS_VIDEO = [
    "green", "purple", "yellow", "pink", "lightgreen",
    "red", "blue1", "blue2", "darkgray", "orange"
]


def _load_bundle_frames(bundle_dir: Path) -> List[Path]:
    return sorted(bundle_dir.glob("*.pkl"))


@contextmanager
def _acquire_lock_with_timeout(lock_file: Path, timeout: float = 60.0):
    """Acquire a directory-based lock with timeout.
    
    Args:
        lock_file: Path to lock directory
        timeout: Maximum time to wait for lock (in seconds)
    
    Yields:
        bool: True if lock was acquired, False if timeout
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    acquired = False
    
    while not acquired:
        try:
            os.makedirs(lock_file, exist_ok=False)
            acquired = True
        except FileExistsError:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                yield False
                return
            time.sleep(0.5)  # Wait 0.5s before retry
    
    try:
        yield True
    finally:
        try:
            lock_file.rmdir()
        except OSError:
            pass


def _load_frame_data(frame_path: Path, *, object_only: bool = False) -> Dict:
    with open(frame_path, "rb") as f:
        frame_data = pickle.load(f)
    if object_only:
        meshes = frame_data.get("meshes")
        if meshes is not None:
            filtered_meshes = [
                mesh for mesh in meshes if "hand" not in mesh.get("name", "").lower()
            ]
            frame_data = dict(frame_data)
            frame_data["meshes"] = filtered_meshes
    return frame_data


def _translate_vertices(vertices: Iterable[Iterable[float]], offset: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    return verts + offset[np.newaxis, :]


def _override_color(
    mesh_name: str,
    original_color: Sequence[float],
    object_color: Tuple[float, float, float],
    vis_contact: bool,
) -> Tuple[List[float], bool]:
    color_arr = np.asarray(original_color, dtype=np.float32)
    alpha = float(color_arr[3]) if color_arr.size >= 4 else 1.0
    name_lower = mesh_name.lower()

    if vis_contact and color_arr.size >= 3 and np.allclose(color_arr[:3], 1.0, atol=1e-3):
        red = COLOR_CHOICES["red"]
        return [red[0], red[1], red[2], alpha], True

    if "left" in name_lower:
        left = COLOR_CHOICES["blue1"]
        return [left[0], left[1], left[2], alpha], True
    if "right" in name_lower:
        right = COLOR_CHOICES["blue2"]
        return [right[0], right[1], right[2], alpha], True
    if "object" in name_lower:
        return [object_color[0], object_color[1], object_color[2], alpha], True

    return list(color_arr[:3]) + [alpha], False


def _prepare_blend_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def _resolve_time_slice_indices(
    token: str | int,
    seq_configs: Sequence[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    frame_lookup: Dict[str, List[Path]],
    seq_offsets: Dict[str, Tuple[int, int]],
) -> Tuple[str, List[int]]:
    if isinstance(token, str):
        normalized = token.strip().lower()
    else:
        normalized = str(int(token))

    use_global = normalized.isdigit()
    indices: List[int] = []
    for seq_obj, _, _, _ in seq_configs:
        paths = frame_lookup.get(seq_obj)
        if not paths:
            raise ValueError(f"No frames available for {seq_obj}")

        if normalized in {"first", "start"}:
            idx = 0
        elif normalized in {"last", "end"}:
            idx = len(paths) - 1
        elif use_global:
            offset, length = seq_offsets.get(seq_obj, (0, len(paths)))
            global_idx = int(normalized)
            local_idx = global_idx - offset
            if local_idx < 0:
                local_idx = 0
            elif local_idx >= length:
                local_idx = length - 1
            idx = local_idx
        else:
            try:
                idx = int(normalized)
            except ValueError as exc:
                raise ValueError(f"Unsupported time-slice token '{token}'") from exc
            idx = max(0, min(idx, len(paths) - 1))
        indices.append(idx)

    if normalized in {"start", "first"}:
        label = "first"
    elif normalized in {"end", "last"}:
        label = "last"
    elif normalized.isdigit():
        label = f"frame{int(normalized):04d}"
    else:
        label = normalized.replace(" ", "_")
    return label, indices


def _build_time_slice_asset(
    seq_configs: Sequence[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    frame_lookup: Dict[str, List[Path]],
    frame_indices: Sequence[int],
    render_hand_flags: Dict[str, bool],
    vis_contact: bool,
    active_seq_obj: str,
) -> Dict:
    aggregated_vertices: List[np.ndarray] = []
    meshes: List[Dict] = []
    camera_payload = None

    for (seq_obj, bundle_dir, offset_vec, object_color), frame_idx in zip(seq_configs, frame_indices):
        paths = frame_lookup[seq_obj]
        if not paths:
            continue
        frame_idx = max(0, min(frame_idx, len(paths) - 1))
        frame_data = _load_frame_data(
            paths[frame_idx],
            object_only=(seq_obj != active_seq_obj or not render_hand_flags.get(seq_obj, True)),
        )

        if camera_payload is None and frame_data.get("camera") is not None:
            camera_info = frame_data["camera"]
            # Camera in bundle is already aligned by wTwp in blender_cvt.py, so use it directly
            camera_payload = {
                "intrinsic": np.asarray(camera_info["intrinsic"], dtype=np.float32),
                "extrinsic_wTc": np.asarray(camera_info["extrinsic_wTc"], dtype=np.float32),
                "width": int(camera_info["width"]),
                "height": int(camera_info["height"]),
            }

        for mesh_info in frame_data.get("meshes", []):
            name_lower = mesh_info.get("name", "").lower()
            if not render_hand_flags.get(seq_obj, True) and "hand" in name_lower:
                continue
            verts = mesh_info.get("vertices")
            if verts is None or len(verts) == 0:
                continue
            translated_vertices = _translate_vertices(verts, offset_vec)
            aggregated_vertices.append(translated_vertices)

            mesh_copy = dict(mesh_info)
            mesh_copy["vertices"] = translated_vertices.tolist()
            mesh_copy["name"] = f"{seq_obj}_{mesh_info['name']}"
            desired_color, force = _override_color(
                mesh_info.get("name", ""),
                mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]),
                object_color,
                vis_contact,
            )
            mesh_copy["color"] = desired_color
            if force:
                mesh_copy["force_color"] = True
            meshes.append(mesh_copy)

    if not aggregated_vertices:
        raise ValueError("No geometry available to build time-slice asset")

    alloc_camera = compute_allocentric_camera_from_assets(aggregated_vertices)
    alloc_payload = {
        "intrinsic": alloc_camera["intrinsic"],
        "extrinsic_wTc": alloc_camera["extrinsic_wTc"],
        "center": alloc_camera["center"],
        "radius": float(alloc_camera["radius"]),
        "width": int(alloc_camera["width"]),
        "height": int(alloc_camera["height"]),
    }

    asset = {
        "meshes": meshes,
        "camera": camera_payload,
        "alloc_camera": alloc_payload,
        "image_path": None,
    }
    return asset


def _render_time_slices(
    seq_configs: Sequence[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    render_hand_flags: Dict[str, bool],
    time_slice_map: Dict[str, List[int]],
    vis_contact: bool,
    vis_obj_trail: bool,
    render_camera: bool,
    blender_exec: Path,
    output_root: Path,
    main_image_dir: Path,
    save_blend: bool,
) -> None:
    if not time_slice_map:
        return

    frame_lookup: Dict[str, List[Path]] = {}
    for seq_obj, bundle_dir, _, _ in seq_configs:
        paths = _load_bundle_frames(bundle_dir)
        if not paths:
            raise ValueError(f"No frames found when preparing time-slice render for {seq_obj}")
        frame_lookup[seq_obj] = paths

    time_slice_root = output_root / "time_slices"
    label_usage: Dict[str, int] = {}

    for seq_obj_active, index_list in time_slice_map.items():
        if not index_list:
            continue
        for idx_pos, raw_idx in enumerate(index_list):
            per_seq_indices: List[int] = []
            for other_seq_obj, _, _, _ in seq_configs:
                paths = frame_lookup[other_seq_obj]
                length = len(paths)
                candidate_indices = time_slice_map.get(other_seq_obj)
                candidate_raw = raw_idx
                if candidate_indices:
                    if idx_pos < len(candidate_indices):
                        candidate_raw = candidate_indices[idx_pos]
                    else:
                        candidate_raw = candidate_indices[-1]
                if candidate_raw == -1:
                    resolved_idx = length - 1
                else:
                    resolved_idx = max(0, min(int(candidate_raw), length - 1))
                per_seq_indices.append(resolved_idx)

            active_idx = per_seq_indices[
                next(i for i, cfg in enumerate(seq_configs) if cfg[0] == seq_obj_active)
            ]
            base_label = f"{seq_obj_active}_frame{active_idx:04d}"
            usage = label_usage.get(base_label, 0)
            label = base_label if usage == 0 else f"{base_label}_{usage+1:02d}"
            label_usage[base_label] = usage + 1

            asset = _build_time_slice_asset(
                seq_configs,
                frame_lookup,
                per_seq_indices,
                render_hand_flags,
                vis_contact,
                active_seq_obj=seq_obj_active,
            )

            slice_root = time_slice_root / label
            slice_bundle_dir = slice_root / "blender_bundle"
            slice_image_dir = slice_root / "images"
            slice_bundle_dir.mkdir(parents=True, exist_ok=True)
            slice_image_dir.mkdir(parents=True, exist_ok=True)

            asset_path = slice_bundle_dir / "0000.pkl"
            with open(asset_path, "wb") as f:
                pickle.dump(asset, f)

            any_hands = any(render_hand_flags.get(seq_obj, True) for seq_obj, _, _, _ in seq_configs)
            blend_path = None
            if save_blend:
                blend_path = _prepare_blend_path(slice_image_dir / f"time_slice_{label}.blend")
            _run_blender(
                blender_exec,
                slice_bundle_dir,
                slice_image_dir,
                object_color="pink",
                hand_colors=DEFAULT_HAND_COLORS,
                target_frame=0,
                allocentric_step=1,
                allocentric_frames=[0],
                render_allocentric=True,
                render_target_frame=False,
                render_camera=render_camera,
                render_hand=any_hands,
                render_obj_trail=vis_obj_trail,
                vis_contact=vis_contact,
                save_blend_path=blend_path,
            )

            allocentric_path = slice_image_dir / "allocentric_overlay.png"
            if allocentric_path.exists():
                renamed = slice_image_dir / f"time_slice_{label}.png"
                os.replace(allocentric_path, renamed)
                main_target = main_image_dir / f"time_slice_{label}.png"
                shutil.copy2(renamed, main_target)


def _check_object_contact(mesh_info: Dict, vis_contact: bool) -> bool:
    """Check if an object mesh is in contact with hands.
    
    Contact is indicated when vis_contact is True and the original color is white (1.0, 1.0, 1.0).
    This matches the logic in _override_color which converts white objects to red when in contact.
    """
    if not vis_contact:
        return False
    if "object" not in mesh_info.get("name", "").lower():
        return False
    
    # Check the original color before any override
    # If the color is white (1.0, 1.0, 1.0), it indicates contact when vis_contact is True
    color = mesh_info.get("color", [])
    if len(color) >= 3:
        color_arr = np.asarray(color[:3], dtype=np.float32)
        # Check if color is white (indicates contact in the original data)
        if np.allclose(color_arr, 1.0, atol=1e-3):
            return True
    
    # Also check if color is red (already been overridden to indicate contact)
    if len(color) >= 3:
        color_arr = np.asarray(color[:3], dtype=np.float32)
        red = COLOR_CHOICES.get("red", [1.0, 0.0, 0.0])
        if np.allclose(color_arr[:3], red, atol=1e-3):
            return True
    
    return False


def _build_combined_bundles(
    seq_configs: List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    render_hand_flags: Dict[str, bool],
    vis_contact: bool,
) -> Tuple[List[Dict], Dict[str, Tuple[int, int]]]:
    per_seq_frames: Dict[str, List[Dict]] = {}
    aggregated_vertices: List[np.ndarray] = []

    # Load frames for all seq_objs
    for seq_obj, bundle_dir, _, _ in seq_configs:
        frame_paths = _load_bundle_frames(bundle_dir)
        frame_data = [_load_frame_data(path) for path in frame_paths]
        if not frame_data:
            raise ValueError(f"No frames found in bundle directory {bundle_dir}")
        per_seq_frames[seq_obj] = frame_data

    # Group seq_objs by sequence ID
    seq_id_to_seq_objs: Dict[str, List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]]] = {}
    for seq_config in seq_configs:
        seq_obj = seq_config[0]
        seq_id = _extract_sequence_id(seq_obj)
        if seq_id not in seq_id_to_seq_objs:
            seq_id_to_seq_objs[seq_id] = []
        seq_id_to_seq_objs[seq_id].append(seq_config)

    combined_frames: List[Dict] = []
    seq_offsets: Dict[str, Tuple[int, int]] = {}
    current_offset = 0

    # Process each sequence ID
    for seq_id, seq_configs_for_seq in seq_id_to_seq_objs.items():
        # Determine max frame count for this sequence
        max_frame_count = max(len(per_seq_frames[seq_obj[0]]) for seq_obj in seq_configs_for_seq)
        
        # Store offsets for this sequence
        for seq_obj, _, _, _ in seq_configs_for_seq:
            frames = per_seq_frames[seq_obj]
            frame_count = len(frames)
            seq_offsets[seq_obj] = (current_offset, frame_count)
        
        # The first seq_obj is used as default for hands and camera
        default_seq_obj, default_bundle_dir, default_offset_vec, _ = seq_configs_for_seq[0]
        
        # Merge frames for each timestep
        for frame_idx in range(max_frame_count):
            new_meshes: List[Dict] = []
            camera_payload = None
            image_path = None
            
            # Collect hands from default seq_obj (or from seq_obj with object in contact)
            default_frame = per_seq_frames[default_seq_obj][min(frame_idx, len(per_seq_frames[default_seq_obj]) - 1)]
            hand_meshes: List[Dict] = []
            contact_seq_obj_for_hands = None
            
            # First pass: check if any object from other seq_objs is in contact
            for seq_obj, bundle_dir, offset_vec, object_color in seq_configs_for_seq:
                if seq_obj == default_seq_obj:
                    continue
                local_idx = min(frame_idx, len(per_seq_frames[seq_obj]) - 1)
                frame_data = per_seq_frames[seq_obj][local_idx]
                
                for mesh_info in frame_data.get("meshes", []):
                    import ipdb; ipdb.set_trace()
                    if "object" in mesh_info.get("name", "").lower():
                        if _check_object_contact(mesh_info, vis_contact):
                            # This object is in contact, use hands from this seq_obj
                            contact_seq_obj_for_hands = seq_obj
                            break
                if contact_seq_obj_for_hands:
                    break
            
            # Get hands from contact seq_obj if found, otherwise from default
            hands_source_seq_obj = contact_seq_obj_for_hands if contact_seq_obj_for_hands else default_seq_obj
            hands_source_frame = per_seq_frames[hands_source_seq_obj][min(frame_idx, len(per_seq_frames[hands_source_seq_obj]) - 1)]
            
            # Extract hands from source
            for mesh_info in hands_source_frame.get("meshes", []):
                name_lower = mesh_info.get("name", "").lower()
                if "hand" in name_lower and render_hand_flags.get(hands_source_seq_obj, True):
                    verts = mesh_info.get("vertices")
                    if verts is not None and len(verts) > 0:
                        # Use offset from default seq_obj for hands
                        translated_vertices = _translate_vertices(verts, default_offset_vec)
                        aggregated_vertices.append(translated_vertices)
                        
                        mesh_copy = dict(mesh_info)
                        mesh_copy["vertices"] = translated_vertices.tolist()
                        mesh_copy["name"] = f"{hands_source_seq_obj}_{mesh_info['name']}"
                        
                        # Color hands according to their names
                        desired_color, force = _override_color(
                            mesh_info.get("name", ""),
                            mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]),
                            (0.8, 0.8, 0.8),  # Dummy object color for hands
                            vis_contact,
                        )
                        mesh_copy["color"] = desired_color
                        if force:
                            mesh_copy["force_color"] = True
                        hand_meshes.append(mesh_copy)
            
            new_meshes.extend(hand_meshes)
            
            # Collect objects from all seq_objs
            for seq_obj, bundle_dir, offset_vec, object_color in seq_configs_for_seq:
                local_idx = min(frame_idx, len(per_seq_frames[seq_obj]) - 1)
                frame_data = per_seq_frames[seq_obj][local_idx]
                
                for mesh_info in frame_data.get("meshes", []):
                    name_lower = mesh_info.get("name", "").lower()
                    if "object" in name_lower:
                        verts = mesh_info.get("vertices")
                        if verts is None or len(verts) == 0:
                            continue
                        translated_vertices = _translate_vertices(verts, offset_vec)
                        aggregated_vertices.append(translated_vertices)
                        
                        mesh_copy = dict(mesh_info)
                        mesh_copy["vertices"] = translated_vertices.tolist()
                        mesh_copy["name"] = f"{seq_obj}_{mesh_info['name']}"
                        
                        desired_color, force = _override_color(
                            mesh_info.get("name", ""),
                            mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]),
                            object_color,
                            vis_contact,
                        )
                        mesh_copy["color"] = desired_color
                        if force:
                            mesh_copy["force_color"] = True
                        new_meshes.append(mesh_copy)
            
            # Use camera from default seq_obj
            if camera_payload is None:
                camera_info = default_frame.get("camera")
                if camera_info is None:
                    raise ValueError(f"Frame from {default_bundle_dir} is missing camera data")
                camera_payload = {
                    "intrinsic": np.asarray(camera_info["intrinsic"], dtype=np.float32),
                    "extrinsic_wTc": np.asarray(camera_info["extrinsic_wTc"], dtype=np.float32),
                    "width": int(camera_info["width"]),
                    "height": int(camera_info["height"]),
                }
            
            # Use image_path from default seq_obj
            if image_path is None:
                default_image_path = default_frame.get("image_path")
                if default_image_path is not None:
                    image_path = str(default_image_path)
            
            combined_frames.append({
                "meshes": new_meshes,
                "camera": camera_payload,
                "image_path": image_path,
            })
        
        current_offset += max_frame_count

    if not aggregated_vertices:
        raise ValueError("No mesh vertices found across the provided sequences")

    alloc_camera = compute_allocentric_camera_from_assets(aggregated_vertices)
    alloc_payload = {
        "intrinsic": alloc_camera["intrinsic"],
        "extrinsic_wTc": alloc_camera["extrinsic_wTc"],
        "center": alloc_camera["center"].tolist(),
        "radius": float(alloc_camera["radius"]),
        "width": int(alloc_camera["width"]),
        "height": int(alloc_camera["height"]),
    }

    for frame in combined_frames:
        frame["alloc_camera"] = alloc_payload

    return combined_frames, seq_offsets


def _parse_per_sequence_targets(spec: str, seq_objs: Sequence[str]) -> Dict[str, List[int]]:
    groups = [segment.strip() for segment in spec.split(",") if segment.strip()]
    if len(groups) != len(seq_objs):
        raise ValueError(
            f"target_frames expects {len(seq_objs)} groups, got {len(groups)}"
        )
    mapping: Dict[str, List[int]] = {}
    for seq_obj, group in zip(seq_objs, groups):
        tokens = [token.strip() for token in group.split("+") if token.strip()]
        if not tokens:
            raise ValueError(f"No frame indices provided for {seq_obj}")
        try:
            mapping[seq_obj] = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"Non-integer frame index in group '{group}'") from exc
    return mapping


def _parse_global_frame_list(spec: Sequence[int] | Sequence[str] | str) -> List[int]:
    if isinstance(spec, str):
        tokens = [token.strip() for token in spec.split(",") if token.strip()]
    else:
        tokens = list(spec)
    if not tokens:
        raise ValueError("No frame indices provided")
    result: List[int] = []
    for token in tokens:
        if isinstance(token, str):
            token = token.strip()
            if not token:
                continue
        try:
            result.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid frame index '{token}'") from exc
    if not result:
        raise ValueError("No valid frame indices parsed")
    return result


def _resolve_frame_selection(
    target_spec: Sequence[int] | Sequence[str] | str | None,
    seq_objs: Sequence[str],
    seq_offsets: Dict[str, Tuple[int, int]],
    total_frames: int,
) -> Tuple[List[int], List[int] | None]:
    if target_spec is None:
        default = [idx for idx in DEFAULT_TARGET_FRAMES if idx < total_frames]
        if default:
            return sorted(default), None
        return list(range(total_frames)), None

    if isinstance(target_spec, str) and "+" in target_spec:
        per_seq = _parse_per_sequence_targets(target_spec, seq_objs)
        global_frames: List[int] = []
        for seq_obj in seq_objs:
            offset, length = seq_offsets.get(seq_obj, (None, None))
            if offset is None or length is None:
                raise KeyError(f"Missing offset metadata for {seq_obj}")
            for local_idx in per_seq[seq_obj]:
                if local_idx < 0 or local_idx >= length:
                    raise ValueError(
                        f"Frame {local_idx} out of range for {seq_obj} (length {length})"
                    )
                global_frames.append(offset + local_idx)
        if not global_frames:
            raise ValueError("No valid frame indices derived from per-sequence specification")
        ordered = sorted(dict.fromkeys(global_frames))
        return ordered, ordered

    global_frames = _parse_global_frame_list(target_spec)
    filtered = [idx for idx in global_frames if 0 <= idx < total_frames]
    if not filtered:
        raise ValueError("No valid frame indices fall within the available frame range")
    ordered = sorted(dict.fromkeys(filtered))
    return ordered, ordered


def build_teaser(
    bundle_root: str,
    seq_objs: Sequence[str],
    *,
    method: str | None = None,
    spacing: float = DEFAULT_SPACING,
    colors: Sequence[str] | None = None,
    output_dir: str,
    target_frames: Sequence[int] | None = None,
    allocentric_step: int = 50,
    blender_exe: str | None = None,
    vis_contact: bool = False,
    render_camera: bool = False,
    image_folder: str = "images",
    render_hand: bool = True,
    no_render_hand: bool = False,
    vis_obj_trail: bool = False,
    render_hand_per_method: Sequence[int] | str | None = None,
    time_slice_frames: Sequence[str] | str | None = None,
    save_blend: bool = False,
) -> None:
    if isinstance(seq_objs, str):
        seq_objs = [item.strip() for item in seq_objs.split(",") if item.strip()]
    if isinstance(colors, str):
        colors = [item.strip() for item in colors.split(",") if item.strip()]

    render_hand_flags: Dict[str, bool] = {seq_obj: bool(render_hand) for seq_obj in seq_objs}
    if isinstance(no_render_hand, str):
        tokens = [token.strip() for token in no_render_hand.split(",") if token.strip()]
        if tokens and len(tokens) != len(seq_objs):
            raise ValueError(
                f"no_render_hand expects {len(seq_objs)} entries, got {len(tokens)}"
            )
        for seq_obj, token in zip(seq_objs, tokens):
            if token not in {"0", "1"}:
                raise ValueError("no_render_hand entries must be 0 or 1")
            render_hand_flags[seq_obj] = token == "1"
    elif isinstance(no_render_hand, (list, tuple)):
        if len(no_render_hand) != len(seq_objs):
            raise ValueError(
                f"no_render_hand expects {len(seq_objs)} entries, got {len(no_render_hand)}"
            )
        for seq_obj, value in zip(seq_objs, no_render_hand):
            render_hand_flags[seq_obj] = bool(value)
    elif isinstance(no_render_hand, bool):
        if no_render_hand:
            for key in render_hand_flags:
                render_hand_flags[key] = False
    elif isinstance(no_render_hand, (int, float)):
        if bool(no_render_hand):
            for key in render_hand_flags:
                render_hand_flags[key] = False
    else:
        raise ValueError("no_render_hand must be bool, comma-separated string, or sequence")

    if render_hand_per_method is not None:
        if isinstance(render_hand_per_method, str):
            tokens = [token.strip() for token in render_hand_per_method.split(",") if token.strip()]
            values = tokens
        else:
            values = list(render_hand_per_method)
        if len(values) != len(seq_objs):
            raise ValueError(
                f"render_hand_per_method expects {len(seq_objs)} entries, got {len(values)}"
            )
        for seq_obj, value in zip(seq_objs, values):
            if isinstance(value, str):
                if value not in {"0", "1"}:
                    raise ValueError("render_hand_per_method entries must be '0' or '1'")
                render_hand_flags[seq_obj] = value == "1"
            else:
                render_hand_flags[seq_obj] = bool(value)

    time_slice_map: Dict[str, List[int]] = {}
    if time_slice_frames is not None:
        if isinstance(time_slice_frames, str):
            groups = [group.strip() for group in time_slice_frames.split(",") if group.strip()]
        else:
            groups = [str(group).strip() for group in time_slice_frames if str(group).strip()]

        if groups and len(groups) != len(seq_objs):
            raise ValueError(
                f"time_slice_frames expects {len(seq_objs)} groups (one per seq_obj), got {len(groups)}"
            )

        for seq_obj, group in zip(seq_objs, groups):
            tokens = [token.strip() for token in group.split("+") if token.strip()]
            if not tokens:
                raise ValueError(f"No time-slice indices provided for {seq_obj}")
            indices: List[int] = []
            for token in tokens:
                lower = token.lower()
                if lower in {"first", "start"}:
                    indices.append(0)
                elif lower in {"last", "end"}:
                    indices.append(-1)
                else:
                    try:
                        indices.append(int(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid time_slice_frames token '{token}' for {seq_obj}."
                        ) from exc
            time_slice_map[seq_obj] = indices

        if time_slice_map and len(time_slice_map) != len(seq_objs):
            missing = [seq for seq in seq_objs if seq not in time_slice_map]
            raise ValueError(
                "time_slice_frames must provide entries for all sequences. Missing: "
                + ", ".join(missing)
            )

    color_sequence = list(colors) if colors is not None else list(DEFAULT_COLORS)
    if not color_sequence:
        raise ValueError("At least one color must be provided")
    if not seq_objs:
        raise ValueError("At least one seq_obj must be specified")

    seq_configs: List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]] = []
    for idx, seq_obj in enumerate(seq_objs):
        root_path = Path(bundle_root)
        candidate_paths = []
        if method:
            candidate_paths.extend(
                [
                    root_path / method / seq_obj / "blender_bundle",
                    root_path / method / seq_obj,
                ]
            )
        candidate_paths.extend(
            [
                root_path / seq_obj / "blender_bundle",
                root_path / seq_obj,
                root_path if len(seq_objs) == 1 else None,
            ]
        )
        bundle_dir = None
        for path in candidate_paths:
            if path is not None and path.is_dir():
                bundle_dir = path
                break
        if bundle_dir is None:
            raise FileNotFoundError(
                f"Unable to locate bundle directory for {seq_obj}. Checked: {[str(p) for p in candidate_paths if p is not None]}"
            )

        offset = np.array([idx * spacing, 0.0, 0.0], dtype=np.float32)
        color_name = color_sequence[idx % len(color_sequence)]
        if color_name not in COLOR_CHOICES:
            raise KeyError(f"Unknown color name '{color_name}'")
        color_rgb = COLOR_CHOICES[color_name]
        seq_configs.append((seq_obj, bundle_dir, offset, color_rgb))

    combined_frames, seq_offsets = _build_combined_bundles(seq_configs, render_hand_flags, vis_contact)

    total_frames = len(combined_frames)
    frame_list, allocentric_frames = _resolve_frame_selection(
        target_frames,
        seq_objs,
        seq_offsets,
        total_frames,
    )

    output_root = Path(output_dir)
    bundle_output = output_root / "blender_bundle"
    image_output = output_root / image_folder
    bundle_output.mkdir(parents=True, exist_ok=True)
    image_output.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in enumerate(combined_frames):
        frame_path = bundle_output / f"{frame_idx:04d}.pkl"
        with open(frame_path, "wb") as f:
            pickle.dump(frame, f)

    blender_exec = blender_exe or os.environ.get("BLENDER_EXE")
    if blender_exec is None:
        blender_exec = str(Path("/move/u/yufeiy2/Package/blender-4.3.2-linux-x64/blender"))
    blender_exec_path = Path(blender_exec)

    blend_save_path = image_output / "scene.blend"
    any_hands = any(render_hand_flags.values())
    _run_blender(
        blender_exec_path,
        bundle_output,
        image_output,
        object_color="pink",
        hand_colors=DEFAULT_HAND_COLORS,
        target_frame=0,
        allocentric_step=allocentric_step,
        allocentric_frames=allocentric_frames,
        render_allocentric=True,
        render_target_frame=False,
        render_camera=render_camera,
        render_hand=any_hands,
        render_obj_trail=vis_obj_trail,
        vis_contact=vis_contact,
        save_blend_path=blend_save_path,
    )

    for tf in frame_list:
        _run_blender(
            blender_exec_path,
            bundle_output,
            image_output,
            object_color="pink",
            hand_colors=DEFAULT_HAND_COLORS,
            target_frame=tf,
            allocentric_step=allocentric_step,
            allocentric_frames=allocentric_frames,
            render_allocentric=False,
            render_target_frame=True,
            render_camera=render_camera,
            render_hand=any_hands,
            render_obj_trail=vis_obj_trail,
            vis_contact=vis_contact,
        )

    if time_slice_map:
        _render_time_slices(
            seq_configs,
            render_hand_flags,
            time_slice_map,
            vis_contact,
            vis_obj_trail,
            render_camera,
            blender_exec_path,
            output_root,
            image_output,
            save_blend,
        )


def _extract_sequence_id(seq_obj: str) -> str:
    """Extract sequence ID from seq_obj (format: seq_id_object_id)."""
    if "_" not in seq_obj:
        raise ValueError(f"seq_obj must contain '_' to separate seq_id and object_id: {seq_obj}")
    parts = seq_obj.split("_", 1)
    return parts[0]


def _extract_object_id(seq_obj: str) -> str:
    """Extract object ID from seq_obj (format: seq_id_object_id)."""
    if "_" not in seq_obj:
        raise ValueError(f"seq_obj must contain '_' to separate seq_id and object_id: {seq_obj}")
    parts = seq_obj.split("_", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _find_nearest_object_pose(
    obj_id: str,
    frame_idx: int,
    seq_obj_to_frames: Dict[str, List[Dict]],
    seq_offsets: Dict[str, Tuple[int, int]],
) -> Optional[Dict]:
    """Find nearest neighbor object pose for an object that doesn't appear in current frame."""
    candidate_seqs = [seq_obj for seq_obj in seq_obj_to_frames.keys() 
                     if _extract_object_id(seq_obj) == obj_id]
    
    if not candidate_seqs:
        return None
    
    best_pose = None
    min_distance = float('inf')
    
    for seq_obj in candidate_seqs:
        offset, length = seq_offsets.get(seq_obj, (0, 0))
        frames = seq_obj_to_frames[seq_obj]
        
        local_idx = frame_idx - offset
        if local_idx < 0:
            local_idx = 0
        elif local_idx >= length:
            local_idx = length - 1
        
        frame_data = frames[local_idx]
        for mesh_info in frame_data.get("meshes", []):
            if "object" in mesh_info.get("name", "").lower():
                distance = abs(local_idx - (frame_idx - offset))
                if distance < min_distance:
                    min_distance = distance
                    best_pose = dict(mesh_info)
                    break
    
    return best_pose


def _build_video_teaser_bundles(
    seq_configs: List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    render_hand_flags: Dict[str, bool],
    vis_contact: bool,
) -> Tuple[List[Dict], Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float, float]]]:
    """Build video teaser bundles with object tracking across clips.
    
    Merges sequences with the same sequence ID into the same timesteps.
    
    Returns:
        combined_frames: List of frame bundles
        seq_offsets: Mapping from seq_obj to (offset, length)
        obj_id_to_color: Mapping from object_id to RGB color tuple
    """
    per_seq_frames: Dict[str, List[Dict]] = {}
    
    # Load frames for all seq_objs
    for seq_obj, bundle_dir, _, _ in seq_configs:
        frame_paths = _load_bundle_frames(bundle_dir)
        frame_data = [_load_frame_data(path) for path in frame_paths]
        if not frame_data:
            raise ValueError(f"No frames found in bundle directory {bundle_dir}")
        per_seq_frames[seq_obj] = frame_data
    
    all_obj_ids = set()
    for seq_obj in per_seq_frames.keys():
        obj_id = _extract_object_id(seq_obj)
        all_obj_ids.add(obj_id)
    
    obj_ids_sorted = sorted(list(all_obj_ids))[:10]
    obj_id_to_color: Dict[str, Tuple[float, float, float]] = {}
    for idx, obj_id in enumerate(obj_ids_sorted):
        color_name = OBJECT_COLORS_VIDEO[idx % len(OBJECT_COLORS_VIDEO)]
        if color_name not in COLOR_CHOICES:
            raise KeyError(f"Unknown color name '{color_name}'")
        obj_id_to_color[obj_id] = COLOR_CHOICES[color_name]
    
    # Group seq_objs by sequence ID for merging at same timestep
    seq_id_to_seq_objs: Dict[str, List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]]] = {}
    for seq_config in seq_configs:
        seq_obj = seq_config[0]
        seq_id = _extract_sequence_id(seq_obj)
        if seq_id not in seq_id_to_seq_objs:
            seq_id_to_seq_objs[seq_id] = []
        seq_id_to_seq_objs[seq_id].append(seq_config)
    
    # PASS 1: Collect ALL vertices from ALL sequences to compute global allocentric camera
    all_aggregated_vertices: List[np.ndarray] = []
    for seq_obj, bundle_dir, offset_vec, _ in seq_configs:
        frames = per_seq_frames[seq_obj]
        for frame_data in frames:
            for mesh_info in frame_data.get("meshes", []):
                name_lower = mesh_info.get("name", "").lower()
                
                if not render_hand_flags.get(seq_obj, True) and "hand" in name_lower:
                    continue
                
                verts = mesh_info.get("vertices")
                if verts is None or len(verts) == 0:
                    continue
                
                translated_vertices = _translate_vertices(verts, offset_vec)
                all_aggregated_vertices.append(translated_vertices)
    
    # Compute GLOBAL allocentric camera from ALL sequences
    if not all_aggregated_vertices:
        raise ValueError("No geometry available to compute allocentric camera")
    
    fixed_alloc_camera = compute_allocentric_camera_from_assets(all_aggregated_vertices)
    fixed_alloc_payload = {
        "intrinsic": fixed_alloc_camera["intrinsic"],
        "extrinsic_wTc": fixed_alloc_camera["extrinsic_wTc"],
        "center": fixed_alloc_camera["center"].tolist(),
        "radius": float(fixed_alloc_camera["radius"]),
        "width": int(fixed_alloc_camera["width"]),
        "height": int(fixed_alloc_camera["height"]),
    }
    
    # PASS 2: Create global frame mapping
    # For each seq_id, determine max frame count and create mapping: seq_id -> local_frame_idx -> global_frame_idx
    seq_id_to_frame_mapping: Dict[str, Dict[int, int]] = {}  # seq_id -> {local_idx: global_idx}
    combined_frames: List[Dict] = []
    seq_offsets: Dict[str, Tuple[int, int]] = {}
    current_global_frame = 0
    
    # Build frame mapping: sequences with same seq_id share global timesteps
    for seq_id, seq_configs_for_seq in seq_id_to_seq_objs.items():
        max_frame_count = max(len(per_seq_frames[seq_obj[0]]) for seq_obj in seq_configs_for_seq)
        
        # Map local frame indices to global frame indices for this sequence
        frame_mapping: Dict[int, int] = {}
        for local_idx in range(max_frame_count):
            frame_mapping[local_idx] = current_global_frame
            current_global_frame += 1
        
        seq_id_to_frame_mapping[seq_id] = frame_mapping
        
        # Store offsets for seq_objs in this sequence
        for seq_obj, _, _, _ in seq_configs_for_seq:
            frames = per_seq_frames[seq_obj]
            frame_count = len(frames)
            seq_offsets[seq_obj] = (frame_mapping[0], frame_count)
    
    total_global_frames = current_global_frame
    
    # PASS 3: Build frames - merge at same timestep, infill across all sequences
    # Create reverse mapping: global_frame_idx -> (seq_id, local_frame_idx)
    global_to_seq_local: Dict[int, List[Tuple[str, int]]] = {}  # global_idx -> [(seq_id, local_idx), ...]
    for seq_id, frame_mapping in seq_id_to_frame_mapping.items():
        for local_idx, global_idx in frame_mapping.items():
            if global_idx not in global_to_seq_local:
                global_to_seq_local[global_idx] = []
            global_to_seq_local[global_idx].append((seq_id, local_idx))
    
    # Get all expected obj_ids from ALL seq_objs
    all_expected_obj_ids = {_extract_object_id(seq_obj) for seq_obj, _, _, _ in seq_configs}
    
    # Process each global frame
    for global_frame_idx in range(total_global_frames):
        
        # STEP 1: MERGE - Collect all objects from seq_objs at this global timestep
        # Find which sequences share this global timestep
        seq_local_pairs = global_to_seq_local.get(global_frame_idx, [])
        merged_objects: Dict[str, Dict] = {}  # obj_id -> {mesh_info, seq_obj, offset_vec}
        
        default_seq_obj = None
        default_bundle_dir = None
        default_offset_vec = None
        default_seq_id = None
        camera_payload = None
        image_path = None
        
        # Collect objects from all sequences at this global timestep
        for seq_id, local_frame_idx in seq_local_pairs:
            seq_configs_for_seq = seq_id_to_seq_objs[seq_id]
            
            # Get default seq_obj for this sequence (first one)
            if default_seq_obj is None:
                default_seq_obj, default_bundle_dir, default_offset_vec, _ = seq_configs_for_seq[0]
            
            # Collect objects from all seq_objs in this sequence at this local timestep
            for seq_obj, bundle_dir, offset_vec, _ in seq_configs_for_seq:
                local_idx = min(local_frame_idx, len(per_seq_frames[seq_obj]) - 1)
                frame_data = per_seq_frames[seq_obj][local_idx]
                
                if camera_payload is None and frame_data.get("camera") is not None:
                    camera_info = frame_data["camera"]
                    camera_payload = {
                        "intrinsic": np.asarray(camera_info["intrinsic"], dtype=np.float32),
                        "extrinsic_wTc": np.asarray(camera_info["extrinsic_wTc"], dtype=np.float32),
                        "width": int(camera_info["width"]),
                        "height": int(camera_info["height"]),
                    }
                
                if image_path is None:
                    frame_image_path = frame_data.get("image_path")
                    if frame_image_path is not None and Path(frame_image_path).exists():
                        image_path = str(frame_image_path)
                    else:
                        fallback_path = Path(f"/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/extract_images-rot90/clip-{seq_id}/{local_idx:04d}.jpg")
                        if fallback_path.exists():
                            image_path = str(fallback_path)
                
                # Merge: Collect objects from current frame
                for mesh_info in frame_data.get("meshes", []):
                    name_lower = mesh_info.get("name", "").lower()
                    if "object" in name_lower:
                        obj_id = _extract_object_id(seq_obj)
                        verts = mesh_info.get("vertices")
                        if verts is None or len(verts) == 0:
                            continue
                        
                        if obj_id not in merged_objects:
                            merged_objects[obj_id] = {
                                "mesh_info": mesh_info,
                                "seq_obj": seq_obj,
                                "offset_vec": offset_vec,
                            }
        
        # STEP 2: Get hands from default sequence
        hand_meshes: List[Dict] = []
        if default_seq_obj is not None:
            default_local_idx = seq_local_pairs[0][1] if seq_local_pairs else 0
            default_frame = per_seq_frames[default_seq_obj][min(default_local_idx, len(per_seq_frames[default_seq_obj]) - 1)]
            
            # Check for contact-based hand selection (per hand individually)
            # Track which seq_obj to use for left hand and right hand separately
            left_hand_source_seq_obj = None
            right_hand_source_seq_obj = None
            
            for seq_id, local_frame_idx in seq_local_pairs:
                seq_configs_for_seq = seq_id_to_seq_objs[seq_id]
                for seq_obj, _, _, _ in seq_configs_for_seq:
                    if seq_obj == default_seq_obj:
                        continue
                    local_idx = min(local_frame_idx, len(per_seq_frames[seq_obj]) - 1)
                    frame_data = per_seq_frames[seq_obj][local_idx]
                    
                    # Check each hand individually for contact
                    for mesh_info in frame_data.get("meshes", []):
                        name_lower = mesh_info.get("name", "").lower()
                        if "left" in name_lower and "hand" in name_lower:
                            # Check if left hand is in contact (white color indicates contact)
                            color = mesh_info.get("color", [])
                            if len(color) >= 3:
                                color_arr = np.asarray(color[:3], dtype=np.float32)
                                if np.allclose(color_arr, 1.0, atol=1e-3):  # White = in contact
                                    left_hand_source_seq_obj = seq_obj
                        elif "right" in name_lower and "hand" in name_lower:
                            # Check if right hand is in contact (white color indicates contact)
                            color = mesh_info.get("color", [])
                            if len(color) >= 3:
                                color_arr = np.asarray(color[:3], dtype=np.float32)
                                if np.allclose(color_arr, 1.0, atol=1e-3):  # White = in contact
                                    right_hand_source_seq_obj = seq_obj
                    
                    # If both hands found, no need to continue
                    if left_hand_source_seq_obj and right_hand_source_seq_obj:
                        break
                if left_hand_source_seq_obj and right_hand_source_seq_obj:
                    break
            
            # Use default seq_obj if no contact found for a particular hand
            left_hand_source_seq_obj = left_hand_source_seq_obj if left_hand_source_seq_obj else default_seq_obj
            right_hand_source_seq_obj = right_hand_source_seq_obj if right_hand_source_seq_obj else default_seq_obj
            
            hands_local_idx = seq_local_pairs[0][1] if seq_local_pairs else 0
            
            # Get left hand from its source
            left_hand_frame = per_seq_frames[left_hand_source_seq_obj][min(hands_local_idx, len(per_seq_frames[left_hand_source_seq_obj]) - 1)]
            # Get right hand from its source
            right_hand_frame = per_seq_frames[right_hand_source_seq_obj][min(hands_local_idx, len(per_seq_frames[right_hand_source_seq_obj]) - 1)]
            
            # Extract left hand
            for mesh_info in left_hand_frame.get("meshes", []):
                name_lower = mesh_info.get("name", "").lower()
                if "left" in name_lower and "hand" in name_lower and render_hand_flags.get(left_hand_source_seq_obj, True):
                    verts = mesh_info.get("vertices")
                    if verts is not None and len(verts) > 0:
                        translated_vertices = _translate_vertices(verts, default_offset_vec)
                        mesh_copy = dict(mesh_info)
                        mesh_copy["vertices"] = translated_vertices.tolist()
                        mesh_copy["name"] = f"{left_hand_source_seq_obj}_{mesh_info['name']}"
                        desired_color, force = _override_color(
                            mesh_info.get("name", ""),
                            mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]),
                            (0.8, 0.8, 0.8),
                            vis_contact,
                        )
                        mesh_copy["color"] = desired_color
                        if force:
                            mesh_copy["force_color"] = True
                        hand_meshes.append(mesh_copy)
            
            # Extract right hand
            for mesh_info in right_hand_frame.get("meshes", []):
                name_lower = mesh_info.get("name", "").lower()
                if "right" in name_lower and "hand" in name_lower and render_hand_flags.get(right_hand_source_seq_obj, True):
                    verts = mesh_info.get("vertices")
                    if verts is not None and len(verts) > 0:
                        translated_vertices = _translate_vertices(verts, default_offset_vec)
                        mesh_copy = dict(mesh_info)
                        mesh_copy["vertices"] = translated_vertices.tolist()
                        mesh_copy["name"] = f"{right_hand_source_seq_obj}_{mesh_info['name']}"
                        desired_color, force = _override_color(
                            mesh_info.get("name", ""),
                            mesh_info.get("color", [0.8, 0.8, 0.8, 1.0]),
                            (0.8, 0.8, 0.8),
                            vis_contact,
                        )
                        mesh_copy["color"] = desired_color
                        if force:
                            mesh_copy["force_color"] = True
                        hand_meshes.append(mesh_copy)
        
        new_meshes: List[Dict] = []
        new_meshes.extend(hand_meshes)
        
        # STEP 3: INFILL - Ensure ALL expected objects appear in EVERY frame
        # Search across ALL sequences and ALL frames for infilling
        for obj_id in all_expected_obj_ids:
            if obj_id in merged_objects:
                # Object found in merged scene - use it
                obj_data = merged_objects[obj_id]
                mesh_info = obj_data["mesh_info"]
                source_offset_vec = obj_data["offset_vec"]
                
                translated_vertices = _translate_vertices(mesh_info["vertices"], source_offset_vec)
                
                mesh_copy = dict(mesh_info)
                mesh_copy["vertices"] = translated_vertices.tolist()
                mesh_copy["name"] = f"{obj_data['seq_obj']}_{mesh_info['name']}"
                
                obj_color = obj_id_to_color.get(obj_id, COLOR_CHOICES["pink"])
                alpha = mesh_info.get("color", [1.0, 1.0, 1.0, 1.0])[3] if len(mesh_info.get("color", [])) > 3 else 1.0
                mesh_copy["color"] = [obj_color[0], obj_color[1], obj_color[2], alpha]
                mesh_copy["force_color"] = True
                new_meshes.append(mesh_copy)
            else:
                # Object missing - infill from nearest frame across ALL sequences
                nearest_pose = None
                nearest_offset_vec = None
                min_distance = float('inf')
                
                # Search ALL seq_objs with same obj_id across ALL frames globally
                for seq_obj, bundle_dir, offset_vec, _ in seq_configs:
                    if _extract_object_id(seq_obj) != obj_id:
                        continue
                    
                    frames = per_seq_frames[seq_obj]
                    # Find which seq_id this seq_obj belongs to
                    seq_obj_seq_id = _extract_sequence_id(seq_obj)
                    
                    # Search all frames of this seq_obj
                    for local_frame_idx, frame_data in enumerate(frames):
                        # Convert to global frame index for distance calculation
                        global_frame_of_local = seq_id_to_frame_mapping.get(seq_obj_seq_id, {}).get(local_frame_idx, global_frame_idx)
                        temporal_distance = abs(global_frame_of_local - global_frame_idx)
                        
                        for mesh_info in frame_data.get("meshes", []):
                            if "object" in mesh_info.get("name", "").lower():
                                if temporal_distance < min_distance:
                                    min_distance = temporal_distance
                                    nearest_pose = dict(mesh_info)
                                    nearest_offset_vec = offset_vec
                                break  # Only check first object mesh per frame
                
                if nearest_pose is not None:
                    # Find the seq_obj that this obj_id belongs to (for naming)
                    source_seq_obj = None
                    for seq_obj, _, _, _ in seq_configs:
                        if _extract_object_id(seq_obj) == obj_id:
                            source_seq_obj = seq_obj
                            break
                    
                    translated_vertices = _translate_vertices(
                        nearest_pose["vertices"], nearest_offset_vec
                    )
                    
                    mesh_copy = dict(nearest_pose)
                    mesh_copy["vertices"] = translated_vertices.tolist()
                    mesh_copy["name"] = f"{source_seq_obj or 'unknown'}_{nearest_pose.get('name', 'object')}"
                    
                    obj_color = obj_id_to_color.get(obj_id, COLOR_CHOICES["pink"])
                    alpha = nearest_pose.get("color", [1.0, 1.0, 1.0, 1.0])[3] if len(nearest_pose.get("color", [])) > 3 else 1.0
                    mesh_copy["color"] = [obj_color[0], obj_color[1], obj_color[2], alpha]
                    mesh_copy["force_color"] = True
                    new_meshes.append(mesh_copy)
        
        # Use camera from default if not set
        if camera_payload is None and default_seq_obj is not None:
            default_local_idx = seq_local_pairs[0][1] if seq_local_pairs else 0
            default_frame = per_seq_frames[default_seq_obj][min(default_local_idx, len(per_seq_frames[default_seq_obj]) - 1)]
            camera_info = default_frame.get("camera")
            if camera_info is None:
                raise ValueError(f"Frame from {default_bundle_dir} is missing camera data")
            camera_payload = {
                "intrinsic": np.asarray(camera_info["intrinsic"], dtype=np.float32),
                "extrinsic_wTc": np.asarray(camera_info["extrinsic_wTc"], dtype=np.float32),
                "width": int(camera_info["width"]),
                "height": int(camera_info["height"]),
            }
        
        if image_path is None and default_seq_obj is not None:
            default_local_idx = seq_local_pairs[0][1] if seq_local_pairs else 0
            default_frame = per_seq_frames[default_seq_obj][min(default_local_idx, len(per_seq_frames[default_seq_obj]) - 1)]
            default_image_path = default_frame.get("image_path")
            if default_image_path is not None:
                image_path = str(default_image_path)
        
        if not new_meshes:
            raise ValueError(f"No geometry available for global frame {global_frame_idx}")
        
        combined_frames.append({
            "meshes": new_meshes,
            "camera": camera_payload,
            "alloc_camera": fixed_alloc_payload,
            "image_path": image_path,
        })
    
    return combined_frames, seq_offsets, obj_id_to_color


def build_teaser_video(
    bundle_root: str,
    seq_objs: Sequence[str] | None = None,
    *,
    pred_dir: str | None = None,
    method: str | None = None,
    spacing: float = DEFAULT_SPACING,
    output_dir: str | None = None,
    blender_exe: str | None = None,
    vis_contact: bool = False,
    image_folder: str = "images",
    render_hand: bool = True,
    no_render_hand: bool = False,
    video_fps: int = 30,
    render_width: int = 1440,
    render_height: int = 1080,
    render_cam_h: int | None = None,
    render_samples: int = 64,
    dynamic_floor: bool = False,
) -> None:
    """Build video version of teaser with object tracking across clips.
    
    Args:
        bundle_root: Root directory containing bundle folders
        seq_objs: List of sequence/object identifiers (format: seq_id_object_id). If None and pred_dir is provided, will discover all items under pred_dir.
        pred_dir: Directory containing pkl files to convert. If provided, will convert to asset list in bundle_root.
        method: Optional method name to look for bundles
        spacing: Spacing between sequences (default: 1.0)
        output_dir: Output directory for rendered videos. If None, uses bundle_root.
        blender_exe: Path to Blender executable
        vis_contact: Visualize contact (default: False)
        image_folder: Image folder name (default: "images")
        render_hand: Render hands (default: True)
        no_render_hand: Skip rendering hands (default: False)
        video_fps: Video FPS (default: 30)
        render_width: Render width (default: 1440)
        render_height: Render height (default: 1080)
        render_cam_h: Camera view height/width (square, default: None uses render_height)
        render_samples: Render samples (default: 64)
        dynamic_floor: Use dynamic floor (default: False)
    """
    # Use bundle_root as output_dir if output_dir is None
    if output_dir is None:
        output_dir = bundle_root
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_dir = output_root / "lock"
    
    # Convert pkl files from pred_dir to asset list in bundle_root if pred_dir is provided
    # Use a single lock for the entire bundle conversion operation
    if pred_dir is not None:
        done_file = lock_dir / "bundle_conversion.done"
        if done_file.exists():
            # Already converted, skip
            pass
        else:
            lock_file = lock_dir / "bundle_conversion.lock"
            with _acquire_lock_with_timeout(lock_file, timeout=60.0) as acquired:
                if acquired:
                    # Double-check after acquiring lock
                    if done_file.exists():
                        # Another process finished while we were waiting for lock
                        pass
                    else:
                        convert_pkl_to_asset_list(pkl_dir=pred_dir, output_dir=bundle_root)
                        # Mark as done
                        done_file.touch()
                # If not acquired (timeout), skip - another process is handling it
    # pinrt out number of meshes in one frame 
    bundle_files = sorted(Path(bundle_root).glob("*.pkl"))
    # Discover seq_objs from pred_dir if not provided
    if seq_objs is None:
        if pred_dir is None:
            raise ValueError("Either seq_objs or pred_dir must be provided")
        
        pred_path = Path(pred_dir)
        discovered_seq_objs: List[str] = []
        
        if pred_path.is_dir():
            # List all .pkl files in directory
            pkl_files = sorted(pred_path.glob("*.pkl"))
            for pkl_file in pkl_files:
                # Try to extract seq_id_object_id from filename or file contents
                index = pkl_file.stem  # filename without extension
                
                # Try to load from file if filename doesn't have the format
                if "_" not in index:
                    try:
                        with open(pkl_file, "rb") as f:
                            pred_data = pickle.load(f)
                            index = pred_data.get("index", index)
                            if isinstance(index, (int, np.integer)):
                                index = pkl_file.stem
                    except Exception:
                        pass
                
                # Extract seq_id_object_id
                if isinstance(index, str) and "_" in index:
                    seq_id, object_id = index.split("_", 1)
                    seq_obj = f"{seq_id}_{object_id}"
                    if seq_obj not in discovered_seq_objs:
                        discovered_seq_objs.append(seq_obj)
                else:
                    # Fallback: use filename as-is if it doesn't match format
                    if index and index not in discovered_seq_objs:
                        discovered_seq_objs.append(index)
        elif pred_path.is_file() and pred_path.suffix == ".pkl":
            # Single file
            index = pred_path.stem
            try:
                with open(pred_path, "rb") as f:
                    pred_data = pickle.load(f)
                    index = pred_data.get("index", index)
                    if isinstance(index, (int, np.integer)):
                        index = pred_path.stem
            except Exception:
                pass
            
            if isinstance(index, str) and "_" in index:
                discovered_seq_objs.append(index)
            elif index:
                discovered_seq_objs.append(index)
        else:
            raise ValueError(f"pred_dir must be a directory or .pkl file: {pred_dir}")
        
        if not discovered_seq_objs:
            raise ValueError(f"No valid seq_objs discovered from pred_dir: {pred_dir}")
        
        seq_objs = discovered_seq_objs
    
    # Parse seq_objs if it's a string
    if isinstance(seq_objs, str):
        seq_objs = [item.strip() for item in seq_objs.split(",") if item.strip()]
    
    if not seq_objs:
        raise ValueError("At least one seq_obj must be specified")
    
    render_hand_flags: Dict[str, bool] = {seq_obj: bool(render_hand) for seq_obj in seq_objs}
    if no_render_hand:
        for key in render_hand_flags:
            render_hand_flags[key] = False
    
    seq_configs: List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]] = []
    for idx, seq_obj in enumerate(seq_objs):
        root_path = Path(bundle_root)
        candidate_paths = []
        if method:
            candidate_paths.extend([
                root_path / method / seq_obj / "blender_bundle",
                root_path / method / seq_obj,
            ])
        candidate_paths.extend([
            root_path / seq_obj / "blender_bundle",
            root_path / seq_obj,
            root_path if len(seq_objs) == 1 else None,
        ])
        
        bundle_dir = None
        for path in candidate_paths:
            if path is not None and path.is_dir():
                bundle_dir = path
                break
        if bundle_dir is None:
            raise FileNotFoundError(
                f"Unable to locate bundle directory for {seq_obj}. "
                f"Checked: {[str(p) for p in candidate_paths if p is not None]}"
            )
        
        offset = np.array([idx * spacing, 0.0, 0.0], dtype=np.float32)
        seq_configs.append((seq_obj, bundle_dir, offset, COLOR_CHOICES["pink"]))
    
    combined_frames, seq_offsets, obj_id_to_color = _build_video_teaser_bundles(
        seq_configs, render_hand_flags, vis_contact
    )
    
    total_frames = len(combined_frames)
    print(f"Built {total_frames} frames for video teaser")
    print(f"Object IDs and colors: {obj_id_to_color}")
    
    output_root = Path(output_dir)
    bundle_output = output_root / "blender_bundle"
    image_output = output_root / image_folder
    lock_dir = output_root / "lock"
    bundle_output.mkdir(parents=True, exist_ok=True)
    image_output.mkdir(parents=True, exist_ok=True)
    
    for frame_idx, frame in enumerate(combined_frames):
        frame_path = bundle_output / f"{frame_idx:04d}.pkl"
        with open(frame_path, "wb") as f:
            pickle.dump(frame, f)
    
    blender_exec = blender_exe or os.environ.get("BLENDER_EXE")
    if blender_exec is None:
        blender_exec = str(Path("/move/u/yufeiy2/Package/blender-4.3.2-linux-x64/blender"))
    blender_exec_path = Path(blender_exec)
    
    any_hands = any(render_hand_flags.values())
    
    print(f"Rendering video for teaser: {total_frames} frames")
    
    video_camera_dir = image_output / "video_camera_frames"
    video_alloc_dir = image_output / "video_alloc_frames"
    video_overlay_dir = image_output / "video_overlay_frames"
    video_camera_dir.mkdir(exist_ok=True)
    video_alloc_dir.mkdir(exist_ok=True)
    video_overlay_dir.mkdir(exist_ok=True)
    
    # Ensure lock directory exists for frame-level locks
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering video frames"):
        camera_frame_path = video_camera_dir / f"{frame_idx:04d}_camera.png"
        alloc_frame_path = video_alloc_dir / f"{frame_idx:04d}_allocentric_overlay.png"
        
        # Check if already done (no lock, output files exist)
        if camera_frame_path.exists() and alloc_frame_path.exists():
            # Already rendered, skip
            continue
        
        # Try to acquire lock
        lock_file = lock_dir / f"frame_{frame_idx:04d}.lock"
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            # Another process is working on this frame, skip
            continue
        
        # Double-check after acquiring lock (race condition protection)
        if camera_frame_path.exists() and alloc_frame_path.exists():
            # Another process finished while we were waiting for lock
            lock_file.rmdir()
            continue
        
        # Do the work
        if not (camera_frame_path.exists() and alloc_frame_path.exists()):
            _run_blender(
                blender_exec_path,
                bundle_output,
                video_camera_dir,
                object_color="pink",
                hand_colors=DEFAULT_HAND_COLORS,
                target_frame=frame_idx,
                allocentric_step=1,
                render_allocentric=False,
                render_target_frame=True,
                render_camera=False,
                render_hand=any_hands,
                render_obj=True,
                render_obj_trail=False,
                vis_contact=vis_contact,
                render_video=True,
                video_frame_idx=frame_idx,
                render_width=render_width,
                render_height=render_height,
                render_cam_h=render_cam_h,
                render_samples=render_samples,
                dynamic_floor=dynamic_floor,
            )
            
            _run_blender(
                blender_exec_path,
                bundle_output,
                video_alloc_dir,
                object_color="pink",
                hand_colors=DEFAULT_HAND_COLORS,
                target_frame=frame_idx,
                allocentric_step=1,
                allocentric_frames=[frame_idx],
                render_allocentric=True,
                render_target_frame=False,
                render_camera=True,
                render_hand=any_hands,
                render_obj=True,
                render_obj_trail=False,
                vis_contact=vis_contact,
                render_video=True,
                video_frame_idx=frame_idx,
                render_width=render_width,
                render_height=render_height,
                render_samples=render_samples,
                dynamic_floor=dynamic_floor,
            )
        
        input_frame_path = video_camera_dir / f"{frame_idx:04d}_input.png"
        input_frame_path_dest = video_overlay_dir / f"{frame_idx:04d}_input.png"
        overlay_frame_path = video_overlay_dir / f"{frame_idx:04d}_overlay.png"
        
        if input_frame_path.exists():
            shutil.copy2(input_frame_path, input_frame_path_dest)
        
        if camera_frame_path.exists() and input_frame_path.exists():
            _overlay_camera_on_input(camera_frame_path, input_frame_path, overlay_frame_path)
        
        lock_file.rmdir()
    
    video_lock_file = lock_dir / "video_creation.lock"
    with _acquire_lock_with_timeout(video_lock_file, timeout=1.0) as acquired:
        if not acquired:
            print("Could not acquire lock for video creation; assuming another process will finish it.")
        else:
            print("Creating videos from frames...")
            alloc_video_path = image_output / "teaser_alloc.mp4"
            _create_video_from_frames(
                video_alloc_dir,
                alloc_video_path,
                fps=video_fps,
                pattern="*_allocentric_overlay.png"
            )
            print(f"Created allocentric video: {alloc_video_path}")

            input_video_path = image_output / "teaser_input.mp4"
            _create_video_from_frames(
                video_overlay_dir,
                input_video_path,
                fps=video_fps,
                pattern="*_input.png"
            )
            print(f"Created input video: {input_video_path}")
            
            camera_video_path = image_output / "teaser_camera.mp4"
            _create_video_from_frames(
                video_camera_dir,
                camera_video_path,
                fps=video_fps,
                pattern="*_camera.png"
            )
            print(f"Created camera video: {camera_video_path}")
            
            overlay_video_path = image_output / "teaser_overlay.mp4"
            _create_video_from_frames(
                video_overlay_dir,
                overlay_video_path,
                fps=video_fps,
                pattern="*_overlay.png"
            )
            print(f"Created overlay video: {overlay_video_path}")
            


if __name__ == "__main__":
    Fire({
        "build_teaser": build_teaser,
        "build_teaser_video": build_teaser_video,
    })
