from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from fire import Fire

from mayday.blender_cvt import compute_allocentric_camera_from_assets
from mayday.blender_wrapper import _run_blender  # reuse existing helper
from mayday.blender_vis  import COLOR_CHOICES

DEFAULT_SPACING = 1.0
DEFAULT_COLORS = [ "green", "purple", "yellow", "pink", "lightgreen"]
DEFAULT_TARGET_FRAMES = [0, 50, 100]
DEFAULT_HAND_COLORS = "blue1,blue2"


def _load_bundle_frames(bundle_dir: Path) -> List[Path]:
    return sorted(bundle_dir.glob("*.pkl"))


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


def _build_combined_bundles(
    seq_configs: List[Tuple[str, Path, np.ndarray, Tuple[float, float, float]]],
    render_hand_flags: Dict[str, bool],
    vis_contact: bool,
) -> Tuple[List[Dict], Dict[str, Tuple[int, int]]]:
    per_seq_frames: Dict[str, List[Dict]] = {}
    aggregated_vertices: List[np.ndarray] = []

    for seq_obj, bundle_dir, _, _ in seq_configs:
        frame_paths = _load_bundle_frames(bundle_dir)
        frame_data = [_load_frame_data(path) for path in frame_paths]
        if not frame_data:
            raise ValueError(f"No frames found in bundle directory {bundle_dir}")
        per_seq_frames[seq_obj] = frame_data

    combined_frames: List[Dict] = []
    seq_offsets: Dict[str, Tuple[int, int]] = {}
    current_offset = 0

    for seq_obj, bundle_dir, offset_vec, object_color in seq_configs:
        frames = per_seq_frames[seq_obj]
        frame_count = len(frames)
        seq_offsets[seq_obj] = (current_offset, frame_count)
        for frame_data in frames:
            meshes = frame_data.get("meshes", [])
            new_meshes: List[Dict] = []
            for mesh_info in meshes:
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
                new_meshes.append(mesh_copy)

            camera_info = frame_data.get("camera")
            if camera_info is None:
                raise ValueError(f"Frame from {bundle_dir} is missing camera data")
            camera_payload = {
                "intrinsic": np.asarray(camera_info["intrinsic"], dtype=np.float32),
                "extrinsic_wTc": np.asarray(camera_info["extrinsic_wTc"], dtype=np.float32),
                "width": int(camera_info["width"]),
                "height": int(camera_info["height"]),
            }

            image_path = frame_data.get("image_path")
            if image_path is not None:
                image_path = str(image_path)

            combined_frames.append(
                {
                    "meshes": new_meshes,
                    "camera": camera_payload,
                    "image_path": image_path,
                }
            )
        current_offset += frame_count

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


if __name__ == "__main__":
    Fire(build_teaser)
