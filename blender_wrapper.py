"""
High-level wrapper that creates Blender asset bundles and renders them.

Output tree:
    outputs/blender_results/<method>/<seq_obj>/
        blender_bundle/{t:04d}.pkl
        images/*.png
"""

from __future__ import annotations

import os.path as osp
import json
from tqdm import tqdm
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Sequence, Tuple, List
from jutils import web_utils
import numpy as np

from mayday.blender_cvt import (
    convert_pkl_to_asset_list,
    compute_allocentric_camera_from_assets,
)
from fire import Fire

LOGGER = logging.getLogger("mayday.blender_wrapper")

SAVE_ROOT = Path("outputs/blender_results")

METHOD_TO_SOURCE = {
    "teaser": "outputs/blender_results/teaser/{seq_obj}.pkl",
    "ours-gen": "outputs/org/ours/sample/{seq_obj}.pkl",
    "gt": "outputs/org/gt/{seq_obj}.pkl",
    "ours": "outputs/org/ours/post/{seq_obj}.pkl",
    "fp_simple": "outputs/org/ab_simple/post/{seq_obj}.pkl",
    "fp_full": "outputs/org/ab_full/post/{seq_obj}.pkl",
    "fp": "outputs/org/fp/{seq_obj}.pkl",
}

METHOD_TO_COLOR = {
    "ours-gen": "blue1",
    "gt": "green",
    "fp": "red",
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
    convert_pkl_to_asset_list(pkl_dir=str(prediction_path), output_dir=str(bundle_dir))


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
    vis_contact: bool = False,
    save_blend_path: Path | None = None,
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
    cmd.append("--vis-contact" if vis_contact else "--no-vis-contact")
    cmd.append("--render-hand" if render_hand else "--no-render-hand")
    cmd.append("--vis-obj-trail" if render_obj_trail else "--no-vis-obj-trail")
    if save_blend_path is not None:
        cmd.extend(["--save-blend", str(save_blend_path)])
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


def _rename_outputs(image_dir: Path, seq_obj: str, method: str, web_dir: Path) -> None:
    web_dir.mkdir(parents=True, exist_ok=True)
    for path in list(image_dir.glob("*.png")):
        stem = path.stem
        suffix = path.suffix
        new_name = f"{seq_obj}_{stem}_{method}{suffix}"
        target_path = web_dir / new_name
        if target_path.exists():
            continue
        shutil.copy2(path, target_path)


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

    base_render_hand = bool(kwargs.get("render_hand", True))
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
            _rename_outputs(image_dir, seq_obj, method, web_dir)

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
                _rename_outputs(image_dir, seq_obj, method, web_dir)


def rebuild_web_gallery(image_folder: str = "images", method_columns: Sequence[str] = ("gt", "fp_simple", "fp_full", "ours")) -> None:
    _build_webpage(image_folder, method_columns)


def main(
    seq_obj: str | None = None,
    method_list: Sequence[str] = ("gt", "fp_simple", "fp_full", "ours"),
    *,
    target_frames: Sequence[int] | None = None,
    dry: bool = False,
    **kwargs,
) -> None:
    image_folder = kwargs.get("image_folder", "images")
    kwargs["image_folder"] = image_folder

    if not dry:
        if seq_obj is None:
            split = kwargs.get("split", "test50obj")
            split_file = osp.join("data/HOT3D-CLIP", "sets", "split.json")
            with open(split_file, "r", encoding="utf-8") as f:
                split_dict = json.load(f)
            seq_obj_list = split_dict[split]
        else:
            seq_obj_list = [seq_obj]

        for seq in tqdm(seq_obj_list):
            render_all_methods(
                seq,
                method_list,
                target_frames=target_frames,
                **kwargs,
            )

    rebuild_web_gallery(image_folder, method_list)



if __name__ == "__main__":
    Fire(main)
