import logging
import json
import os
import os.path as osp
import pickle
from glob import glob
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from jutils import geom_utils, mesh_utils
from fire import Fire

from egorecon.utils.motion_repr import HandWrapper
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer

LOGGER = logging.getLogger("mayday.blender_cvt")


def _ensure_logger():
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _pose9_to_matrix(pose9: np.ndarray) -> np.ndarray:
    return geom_utils.se3_to_matrix_v2(torch.FloatTensor(pose9)).cpu().numpy()


def _load_dataset_contact(dataset_contact_path: str) -> Dict[str, Dict]:
    with open(dataset_contact_path, "rb") as f:
        data = pickle.load(f)
    return data


def _resolve_ids(index: str, pkl_path: str) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(index, bytes):
        index = index.decode("utf-8")

    if isinstance(index, (int, np.integer)):
        LOGGER.warning(
            "Prediction index '%s' is numeric; cannot infer seq/object IDs directly. "
            "Falling back to filename '%s'.",
            index,
            osp.basename(pkl_path),
        )
        index = osp.splitext(osp.basename(pkl_path))[0]

    if isinstance(index, str) and "_" in index:
        seq_id, object_id = index.split("_", 1)
        return seq_id, object_id

    LOGGER.error(
        "Unable to resolve sequence/object identifiers for '%s'. "
        "Expected index format 'seq_object'.",
        osp.basename(pkl_path),
    )
    return None, None

width, height = 1440, 1080
fov = np.deg2rad(40.0)
az = np.deg2rad(-90.0)
el = np.deg2rad(45.0)

def compute_allocentric_camera_from_assets(
    vertices_list: Sequence[np.ndarray],
    azimuth_deg: float = -90.0,
    elevation_deg: float = 45.0,
    safety_scale: float = .55,
    width: int = 1440,
    height: int = 1080,
    fov_deg: float = 40.0,
) -> Dict[str, np.ndarray]:
    """Compute an allocentric camera that sees all provided meshes.

    Args:
        vertices_list: Iterable of arrays shaped (N, 3) in world coordinates.
        azimuth_deg: Camera azimuth in degrees (PyTorch3D convention).
        elevation_deg: Camera elevation in degrees (PyTorch3D convention).
        safety_scale: Scalar multiplier applied to the bounding radius.
        width: Output image width in pixels.
        height: Output image height in pixels.
        fov_deg: Horizontal field-of-view in degrees (assumed symmetric).

    Returns:
        Dictionary with `intrinsic`, `extrinsic_wTc`, `center`, `radius`, `width`, `height`.
    """
    verts = [np.asarray(v, dtype=np.float32) for v in vertices_list if v is not None]
    if not verts:
        raise ValueError("vertices_list must contain at least one non-empty array.")

    stacked = np.concatenate(verts, axis=0)
    center = stacked.mean(axis=0)

    distances = np.linalg.norm(stacked - center[None, :], axis=1)
    radius = float(distances.max())
    if radius < 1e-6:
        radius = 1.0
    radius *= float(safety_scale)

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    direction = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float32,
    )

    fov_x = np.deg2rad(fov_deg)
    aspect = width / max(height, 1)
    fov_y = 2.0 * np.arctan(np.tan(fov_x / 2.0) / max(aspect, 1e-6))
    sin_x = np.sin(max(fov_x / 2.0, 1e-6))
    sin_y = np.sin(max(fov_y / 2.0, 1e-6))
    min_sin = max(min(sin_x, sin_y), 1e-6)
    distance = radius / min_sin

    cam_pos = center + distance * direction

    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_cam = center - cam_pos
    z_norm = np.linalg.norm(z_cam)
    if z_norm < 1e-6:
        z_cam = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        z_cam = z_cam / z_norm

    x_left = -np.cross(up_world, z_cam)
    x_norm = np.linalg.norm(x_left)
    if x_norm < 1e-6:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_left = -np.cross(up_world, z_cam)
        x_norm = np.linalg.norm(x_left)
    x_left = x_left / x_norm
    y_up = np.cross(z_cam, x_left)
    y_up = y_up / np.linalg.norm(y_up)

    wTc = np.eye(4, dtype=np.float32)
    wTc[:3, 0] = x_left
    wTc[:3, 1] = y_up
    wTc[:3, 2] = z_cam
    wTc[:3, 3] = cam_pos.astype(np.float32)

    fov = np.deg2rad(fov_deg)
    fx = fy = 0.5 * height / np.tan(fov / 2.0)
    intrinsic = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return {
        "intrinsic": intrinsic,
        "extrinsic_wTc": wTc,
        "center": center.astype(np.float32),
        "radius": float(radius),
        "width": int(width),
        "height": int(height),
    }


def compute_allocentric_camera(wTc_mats: List[np.ndarray]) -> Dict[str, np.ndarray]: 
    """

    :param wTc_mats: (T, )
    :raises ValueError: _description_
    :return: _description_
    """
    # compute based on allocentric since all of them uses the same camera.
    if not wTc_mats:
        raise ValueError("No camera matrices provided for allocentric computation.")

    positions = np.stack([mat[:3, 3] for mat in wTc_mats], axis=0)
    center = positions.mean(axis=0)
    center[..., 2] -= 0.3
    max_radius = np.linalg.norm(positions - center, axis=1).max()
    if max_radius < 1e-6:
        max_radius = 1.0
    radius = max_radius * 3

    direction = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float32,
    )
    cam_pos = center + radius * direction

    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_cam = center - cam_pos
    z_norm = np.linalg.norm(z_cam)
    if z_norm < 1e-6:
        z_cam = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        z_cam = z_cam / z_norm

    x_left = -np.cross(up_world, z_cam)
    x_norm = np.linalg.norm(x_left)
    if x_norm < 1e-6:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_left = -np.cross(up_world, z_cam)
        x_norm = np.linalg.norm(x_left)
    x_left = x_left / x_norm
    y_up = np.cross(z_cam, x_left)
    y_up = y_up / np.linalg.norm(y_up)

    wTc = np.eye(4, dtype=np.float32)
    wTc[:3, 0] = x_left
    wTc[:3, 1] = y_up
    wTc[:3, 2] = z_cam
    wTc[:3, 3] = cam_pos.astype(np.float32)

    fx = fy = 0.5 * height / np.tan(fov / 2.0)
    intr = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return {
        "intrinsic": intr,
        "extrinsic_wTc": wTc,
        "center": center.astype(np.float32),
        "radius": float(radius),
        "width": int(width),
        "height": int(height),
    }


def convert_pkl_to_asset_list(
    pkl_dir: str,
    output_dir: str,
    mano_dir: str = "assets/mano",
    object_mesh_dir: str = "data/HOT3D-CLIP/object_models_eval/",
    dataset_contact_path: str = "data/HOT3D-CLIP/preprocess/dataset_contact.pkl",
    contact_threshold: float = 0.5,
) -> None:
    _ensure_logger()

    pkl_dir = osp.abspath(pkl_dir)
    output_dir = osp.abspath(output_dir)
    mano_dir = osp.abspath(mano_dir)
    object_mesh_dir = osp.abspath(object_mesh_dir)
    dataset_contact_path = osp.abspath(dataset_contact_path)

    if osp.isdir(pkl_dir):
        pkl_list = sorted(glob(osp.join(pkl_dir, "*.pkl")))
    else:
        assert osp.isfile(pkl_dir), f"File {pkl_dir} not found"
        pkl_list = [pkl_dir]

    LOGGER.info("Loading dataset contact metadata from %s", dataset_contact_path)
    contact_meta = _load_dataset_contact(dataset_contact_path)
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.info("Initialising MANO hand wrapper from %s", mano_dir)
    hand_wrapper = HandWrapper(mano_dir).cpu()

    LOGGER.info("Loading object mesh library from %s", object_mesh_dir)
    object_library = Pt3dVisualizer.setup_template(
        object_mesh_dir, lib="hotclip", load_mesh=True
    )

    IMAGE_ROOT = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/extract_images-rot90"

    camera_mats = []
    asset_paths = []
    scene_vertices: List[np.ndarray] = []

    for pkl_file in pkl_list:
        with open(pkl_file, "rb") as f:
            pred_file = pickle.load(f)
        for k, v in pred_file.items():
            if hasattr(v, 'shape'):
                if k != 'wTc':
                    pred_file[k] = v[0]


        index = pred_file.get("index", osp.splitext(osp.basename(pkl_file))[0])
        seq_id, object_id = _resolve_ids(index, pkl_file)
        # seq_id, object_id = pkl_file.split('/')[-2].split('.')[0].split('_')
        # index = f"{seq_id}_{object_id}"
        # if seq_id is None or object_id is None:
        #     LOGGER.error("Skipping %s due to unresolved IDs", pkl_file)
        #     continue

        meta = contact_meta.get(seq_id)
        if meta is None:
            LOGGER.error(
                "Sequence %s not found in dataset contact metadata. Skipping %s.",
                seq_id,
                pkl_file,
            )
            continue

        intr = meta.get("intrinsic")

        object_id_key = object_id
        # if object_id_key not in object_library and object_id_key.zfill(6) in object_library:
        #     object_id_key = object_id_key.zfill(6)

        wTo_raw = np.asarray(pred_file["wTo"], dtype=np.float32)
        left_params = torch.as_tensor(pred_file["left_hand_params"], dtype=torch.float32)
        right_params = torch.as_tensor(pred_file["right_hand_params"], dtype=torch.float32)

        if wTo_raw.shape[-1] > 9:
            LOGGER.debug(
                "wTo last dimension %d > 9, truncating to first 9 entries.", wTo_raw.shape[-1]
            )
            wTo_raw = wTo_raw[..., :9]

        wTo_mats = _pose9_to_matrix(wTo_raw)
        frame_count = wTo_mats.shape[0]

        contact = np.asarray(pred_file["contact"], dtype=np.float32)
        if contact.ndim != 2:
            raise ValueError(
                f"Contact tensor must be 2D (T, D); got shape {contact.shape} for {index}."
            )
        if contact.shape[0] != frame_count:
            raise ValueError(
                f"Contact sequence length {contact.shape[0]} does not match "
                f"pose length {frame_count} for {seq_id}_{object_id}."
            )

        wTc = np.asarray(pred_file["wTc"], dtype=np.float32)
        if wTc.shape[-1] == 9:
            wTc = _pose9_to_matrix(wTc)
        camera_mats.extend([mat.astype(np.float32) for mat in wTc])

        wTwp = torch.eye(4, dtype=torch.float32)[None].repeat(frame_count, 1, 1)
        gt_wTc_meta = meta.get("wTc")
        gt_wTc_tensor = torch.as_tensor(
            gt_wTc_meta[0], dtype=torch.float32
        )
        pred_wTc_tensor = torch.as_tensor(wTc[0], dtype=torch.float32)
        wTwp_single = gt_wTc_tensor @ geom_utils.inverse_rt_v2(pred_wTc_tensor)
        wTwp = wTwp_single[None].repeat(frame_count, 1, 1)

        with torch.no_grad():
            left_verts, left_faces, _ = hand_wrapper.hand_para2verts_faces_joints(
                left_params, side="left"
            )
            right_verts, right_faces, _ = hand_wrapper.hand_para2verts_faces_joints(
                right_params, side="right"
            )

        left_verts = mesh_utils.apply_transform(left_verts, wTwp)
        right_verts = mesh_utils.apply_transform(right_verts, wTwp)
        wTo_aligned = torch.matmul(
            wTwp, torch.as_tensor(wTo_mats, dtype=torch.float32)
        )
        wTc_aligned = torch.matmul(wTwp, torch.as_tensor(wTc, dtype=torch.float32))

        left_faces_np = left_faces[0].cpu().numpy().astype(np.int32)
        right_faces_np = right_faces[0].cpu().numpy().astype(np.int32)

        object_mesh = object_library[object_id_key]
        obj_verts_exp = object_mesh.verts_padded().repeat(frame_count, 1, 1).to(
            wTo_aligned.device
        )
        obj_faces = object_mesh.faces_padded()[0].cpu().numpy().astype(np.int32)
        obj_world = mesh_utils.apply_transform(obj_verts_exp, wTo_aligned)

        left_verts = left_verts.cpu().numpy().astype(np.float32)
        right_verts = right_verts.cpu().numpy().astype(np.float32)
        obj_world = obj_world.cpu().numpy().astype(np.float32)
        wTc = wTc_aligned.cpu().numpy().astype(np.float32)

        scene_vertices.extend(
            [
                left_verts.reshape(-1, 3),
                right_verts.reshape(-1, 3),
                obj_world.reshape(-1, 3),
            ]
        )
        scene_vertices.append(wTc[:, :3, 3])

        print(pkl_dir, osp.isdir(pkl_dir))
        if osp.isdir(pkl_dir):
            seq_output_dir = osp.join(output_dir, str(index))
        else:
            seq_output_dir = output_dir

        # seq_output_dir = osp.join(output_dir, str(index))
        os.makedirs(seq_output_dir, exist_ok=True)

        width = int(round(float(intr[0, 2]) * 2))
        height = int(round(float(intr[1, 2]) * 2))

        for t in range(frame_count):
            left_color = [0.8, 0.2, 0.2, 1.0]
            right_color = [0.2, 0.2, 0.8, 1.0]
            object_color = [0.85, 0.85, 0.85, 1.0]

            if contact.size and contact.shape[-1] >= 2:
                if contact[t, 0] > contact_threshold:
                    left_color = [1.0, 1., 1., 1.0]
                if contact[t, 1] > contact_threshold:
                    right_color = [1.0, 1., 1., 1.0]
                if (contact[t] > contact_threshold).any():
                    object_color = [1.0, 1., 1., 1.0]

            image_path = None
            image_path = osp.join(
                IMAGE_ROOT,
                f"clip-{seq_id}",
                f"{t:04d}.jpg",
            )

            asset_bundle = {
                "frame": int(t),
                "seq_id": seq_id,
                "object_id": object_id,
                "prediction_index": index,
                "camera": {
                    "intrinsic": intr.astype(np.float32),
                    "extrinsic_wTc": wTc[t].astype(np.float32) if wTc.size else None,
                    "width": width,
                    "height": height,
                },
                "meshes": [
                    {
                        "name": "left_hand",
                        "vertices": left_verts[t],
                        "faces": left_faces_np,
                        "color": left_color,
                    },
                    {
                        "name": "right_hand",
                        "vertices": right_verts[t],
                        "faces": right_faces_np,
                        "color": right_color,
                    },
                    {
                        "name": "object",
                        "vertices": obj_world[t],
                        "faces": obj_faces,
                        "color": object_color,
                    },
                ],
            }
            assert osp.exists(image_path), image_path
            asset_bundle["image_path"] = image_path

            asset_path = osp.join(seq_output_dir, f"{t:04d}.pkl")
            with open(asset_path, "wb") as f:
                pickle.dump(asset_bundle, f)
            asset_paths.append(asset_path)

        LOGGER.info(
            "Converted %s -> %s (%d frames)",
            osp.basename(pkl_file),
            seq_output_dir,
            frame_count,
        )

    alloc_camera = compute_allocentric_camera_from_assets(scene_vertices)
    alloc_camera["center"] = alloc_camera["center"].tolist()
    for asset_path in asset_paths:
        with open(asset_path, "rb") as f:
            bundle = pickle.load(f)
        bundle["alloc_camera"] = alloc_camera
        with open(asset_path, "wb") as f:
            pickle.dump(bundle, f)


def make_one_web_page_compare():
    LOGGER.warning("make_one_web_page_compare is not implemented yet.")
    return


def covnert_gt_to_pkl(
    save_pkl_dir: str = "outputs/org/gt/",
    split: str = "test50obj",
    data_dir: str = "data/HOT3D-CLIP",
):
    _ensure_logger()

    data_dir = osp.abspath(data_dir)
    dataset_path = osp.join(data_dir, "preprocess", "dataset_contact.pkl")
    split_path = osp.join(data_dir, "sets", "split.json")

    LOGGER.info("Loading ground-truth metadata from %s", dataset_path)
    with open(dataset_path, "rb") as f:
        dataset_contact = pickle.load(f)

    LOGGER.info("Loading split definition from %s", split_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    if split not in split_dict:
        raise KeyError(f"Split '{split}' not found in {split_path}.")

    seq_obj_list = split_dict[split]
    save_pkl_dir = osp.abspath(save_pkl_dir)
    os.makedirs(save_pkl_dir, exist_ok=True)

    for seq_obj in seq_obj_list:
        if "_" not in seq_obj:
            raise ValueError(f"Split entry '{seq_obj}' must be of the form 'seq_object'.")
        seq_id, obj_id = seq_obj.split("_", 1)

        meta = dataset_contact[seq_id]

        wTc = np.asarray(meta["wTc"], dtype=np.float32)
        wTc_param = geom_utils.matrix_to_se3_v2(torch.from_numpy(wTc)).cpu().numpy().astype(np.float32)
        wTo_key = f"obj_{obj_id}_wTo"
        contact_key = f"obj_{obj_id}_contact_lr"

        wTo_mats = np.asarray(meta[wTo_key], dtype=np.float32)
        contact = np.asarray(meta[contact_key], dtype=np.float32)

        if wTo_mats.ndim != 3 or wTo_mats.shape[1:] != (4, 4):
            raise ValueError(f"{wTo_key} for {seq_obj} must be (T,4,4); got shape {wTo_mats.shape}.")
        if contact.ndim != 2 or contact.shape[1] != 2:
            raise ValueError(f"{contact_key} for {seq_obj} must be (T,2); got shape {contact.shape}.")

        left_theta = np.asarray(meta["left_hand_theta"], dtype=np.float32)
        right_theta = np.asarray(meta["right_hand_theta"], dtype=np.float32)
        left_shape = np.asarray(meta["left_hand_shape"], dtype=np.float32)
        right_shape = np.asarray(meta["right_hand_shape"], dtype=np.float32)

        frame_count = wTo_mats.shape[0]
        if contact.shape[0] != frame_count:
            raise ValueError(
                f"Contact length {contact.shape[0]} does not match wTo length {frame_count} for {seq_obj}."
            )

        left_hand_params = np.concatenate([left_theta, left_shape], axis=-1)
        right_hand_params = np.concatenate([right_theta, right_shape], axis=-1)

        wTo_param = (
            geom_utils.matrix_to_se3_v2(torch.from_numpy(wTo_mats)).cpu().numpy()
        )

        output_dict = {
            "index": seq_obj,
            "wTo": wTo_param[None, ...].astype(np.float32),
            "wTc": wTc_param.astype(np.float32),
            "left_hand_params": left_hand_params[None, ...].astype(np.float32),
            "right_hand_params": right_hand_params[None, ...].astype(np.float32),
            "contact": contact[None, ...].astype(np.float32),
        }

        output_path = osp.join(save_pkl_dir, f"{seq_obj}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)

        LOGGER.info("Saved ground-truth pkl to %s", output_path)

def main(mode, **kwargs):
    if mode == "convert":
        convert_pkl_to_asset_list(**kwargs)
    elif mode == "cvt_gt":
        covnert_gt_to_pkl(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    Fire(main)