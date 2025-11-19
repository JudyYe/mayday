"""merge videos
two usage cases: 
"""
import logging
import shutil
from pathlib import Path
from typing import Sequence

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from mayday.blender_wrapper import SAVE_ROOT, _create_video_from_frames

LOGGER = logging.getLogger("mayday.blender_merge")


def _ensure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def merge_videos_3way(
    seq_obj: str,
    method: str,
    image_folder: str = "images",
    video_fps: int = 30,
) -> None:
    """Merge input, camera, and alloc videos into a 3-way layout.
    
    Layout:
    - Top left: input video (H/2 x W/2)
    - Bottom left: camera video (H/2 x W/2)
    - Right: alloc video (H x W)
    
    Args:
        seq_obj: Sequence/object identifier
        method: Method name
        image_folder: Image folder name (default: "images")
        video_fps: Video FPS (default: 30)
    """
    _ensure_logger()
    
    image_dir = SAVE_ROOT / method / seq_obj / image_folder
    video_camera_dir = image_dir / "video_camera_frames"
    
    # Paths to input videos/images
    input_video_path = image_dir / f"{seq_obj}_{method}_input.mp4"
    camera_video_path = image_dir / f"{seq_obj}_{method}_camera.mp4"
    alloc_video_path = image_dir / f"{seq_obj}_{method}_alloc.mp4"
    output_video_path = image_dir / f"{seq_obj}_{method}_mergeto3.mp4"
    if output_video_path.exists():
        LOGGER.info(f"Merged video already exists, skipping: {output_video_path}")
        if method == "ours":
            ways3_dir = SAVE_ROOT / "ours" / "3ways"
            ways3_dir.mkdir(parents=True, exist_ok=True)
            ways3_video_path = ways3_dir / f"{seq_obj}_3ways.mp4"
            if not ways3_video_path.exists():
                shutil.copy2(output_video_path, ways3_video_path)
                LOGGER.info(f"Copied existing merged video to: {ways3_video_path}")
        return
    
    # Check if required files exist
    if not camera_video_path.exists():
        LOGGER.warning(f"Camera video not found: {camera_video_path}")
        return
    if not alloc_video_path.exists():
        LOGGER.warning(f"Alloc video not found: {alloc_video_path}")
        return
    
    # Always create input video from input images (encode from PNG frames)
    LOGGER.info(f"Creating input video from frames in: {video_camera_dir}")
    if not video_camera_dir.exists():
        LOGGER.error(f"Video camera frames directory not found: {video_camera_dir}")
        return
    
    # Check if input frames exist
    input_frame_paths = sorted(video_camera_dir.glob("*_input.png"))
    if not input_frame_paths:
        LOGGER.error(f"No input frames found matching '*_input.png' in {video_camera_dir}")
        return
    
    LOGGER.info(f"Found {len(input_frame_paths)} input frames")
    
    # Create input video from frames
    _create_video_from_frames(
        video_camera_dir,
        input_video_path,
        fps=video_fps,
        pattern="*_input.png"
    )
    
    # Verify input video was created
    if not input_video_path.exists():
        LOGGER.error(f"Failed to create input video: {input_video_path}")
        return
    
    LOGGER.info(f"Input video created: {input_video_path}")
    
    # Read all three videos
    LOGGER.info(f"Reading videos: input={input_video_path}, camera={camera_video_path}, alloc={alloc_video_path}")
    
    input_reader = imageio.get_reader(str(input_video_path))
    camera_reader = imageio.get_reader(str(camera_video_path))
    alloc_reader = imageio.get_reader(str(alloc_video_path))
    
    # Get alloc video dimensions (this will be the full size)
    alloc_meta = alloc_reader.get_meta_data()
    alloc_height = alloc_meta['size'][1]
    alloc_width = alloc_meta['size'][0]
    
    # Input and camera should be half size
    small_height = alloc_height // 2
    small_width = small_height
    
    # Total output dimensions: (H, W + W/2) = (H, 3W/2)
    output_height = alloc_height
    output_width = alloc_width + small_width
    
    LOGGER.info(f"Output dimensions: {output_width}x{output_height}")
    LOGGER.info(f"Small videos (input/camera): {small_width}x{small_height}")
    LOGGER.info(f"Alloc video: {alloc_width}x{alloc_height}")
    
    # Get frame counts
    num_frames = min(
        input_reader.count_frames(),
        camera_reader.count_frames(),
        alloc_reader.count_frames()
    )
    
    LOGGER.info(f"Merging {num_frames} frames...")
    
    # Create output video writer
    output_writer = imageio.get_writer(
        str(output_video_path),
        fps=video_fps,
        codec="libx264",
        quality=8
    )
    
    try:
        for frame_idx in tqdm(range(num_frames), desc="Merging frames"):
            # Read frames
            input_frame = input_reader.get_data(frame_idx)
            camera_frame = camera_reader.get_data(frame_idx)
            alloc_frame = alloc_reader.get_data(frame_idx)
            
            # Resize input and camera to small size
            input_img = Image.fromarray(input_frame)
            camera_img = Image.fromarray(camera_frame)
            
            input_small = input_img.resize((small_width, small_height), Image.Resampling.LANCZOS)
            camera_small = camera_img.resize((small_width, small_height), Image.Resampling.LANCZOS)
            
            # Convert back to numpy
            input_small_arr = np.array(input_small)
            camera_small_arr = np.array(camera_small)
            
            # Create composite frame
            composite = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Top left: input
            composite[:small_height, :small_width] = input_small_arr[:, :, :3]
            
            # Bottom left: camera
            composite[small_height:, :small_width] = camera_small_arr[:, :, :3]
            
            # Right: alloc (full size)
            composite[:, small_width:] = alloc_frame[:, :, :3]
            
            # Write frame
            output_writer.append_data(composite)
    
    finally:
        input_reader.close()
        camera_reader.close()
        alloc_reader.close()
        output_writer.close()
    
    LOGGER.info(f"Created merged video: {output_video_path}")
    
    # Copy to 3ways directory
    if method == "ours":
        ways3_dir = SAVE_ROOT / "ours" / "3ways"
        ways3_dir.mkdir(parents=True, exist_ok=True)
        ways3_video_path = ways3_dir / f"{seq_obj}_3ways.mp4"
        shutil.copy2(output_video_path, ways3_video_path)
        LOGGER.info(f"Copied merged video to: {ways3_video_path}")


def merge_videos_cmp_hoi_aligned(
    seq_obj: str,
    alloc_path: Path,
    method_list: Sequence[str],
    image_folder: str = "images",
    video_fps: int = 30,
) -> None:
    """Merge videos into a 2x4 aligned layout for HOI comparison.
    
    Layout:
    Top row (square videos): input | fp_simple_camera | fp_contact_camera | ours_camera
    Bottom row (square videos cropped from alloc renders):
        gt_alloc | fp_simple_alloc | fp_contact_alloc | ours_alloc
    
    Args:
        seq_obj: Sequence/object identifier.
        alloc_path: Base directory containing method folders with alloc videos.
        method_list: List of method names for camera videos (order matters).
        image_folder: Folder containing camera frames/videos.
        video_fps: Output video FPS.
    """
    _ensure_logger()
    
    # Paths for GT input / camera frames
    gt_image_dir = SAVE_ROOT / "gt" / seq_obj / image_folder
    gt_video_camera_dir = gt_image_dir / "video_camera_frames"
    gt_input_path = gt_image_dir / f"{seq_obj}_gt_input.mp4"
    
    # Build camera video paths for methods
    camera_paths = [
        SAVE_ROOT / method / seq_obj / image_folder / f"{seq_obj}_{method}_camera.mp4"
        for method in method_list
    ]
    
    # Build alloc video paths (gt + each method) from <alloc_path>/<method>/<seq_obj>/video/*.mp4
    alloc_methods = ["gt", *method_list]
    alloc_paths = [
        alloc_path / method / seq_obj / "video" / f"{seq_obj}_{method}_alloc.mp4"
        for method in alloc_methods
    ]
    
    output_path = SAVE_ROOT / "ours" / "cmp_hoi" / f"{seq_obj}_cmp_hoi_aligned.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        LOGGER.info(f"Aligned comparison video already exists, skipping: {output_path}")
        return
    
    # Ensure GT input video exists (encode from frames if needed)
    if not gt_input_path.exists():
        LOGGER.info(f"Creating GT input video from frames in: {gt_video_camera_dir}")
        if not gt_video_camera_dir.exists():
            LOGGER.error(f"GT video camera frames directory not found: {gt_video_camera_dir}")
            return
        input_frame_paths = sorted(gt_video_camera_dir.glob("*_input.png"))
        if not input_frame_paths:
            LOGGER.error(f"No input frames found matching '*_input.png' in {gt_video_camera_dir}")
            return
        _create_video_from_frames(
            gt_video_camera_dir,
            gt_input_path,
            fps=video_fps,
            pattern="*_input.png",
        )
        if not gt_input_path.exists():
            LOGGER.error(f"Failed to create GT input video: {gt_input_path}")
            return
    
    # Validate video availability
    if not gt_input_path.exists():
        LOGGER.warning(f"GT input video not found: {gt_input_path}")
        return
    
    missing_camera = [
        (method, path) for method, path in zip(method_list, camera_paths) if not path.exists()
    ]
    if missing_camera:
        for method, path in missing_camera:
            LOGGER.warning(f"Camera video not found for {method}: {path}")
        return
    
    missing_alloc = [
        (method, path) for method, path in zip(alloc_methods, alloc_paths) if not path.exists()
    ]
    if missing_alloc:
        for method, path in missing_alloc:
            LOGGER.warning(f"Alloc video not found for {method}: {path}")
        return
    
    LOGGER.info(f"Merging aligned HOI videos for {seq_obj}")
    
    # Prepare readers
    gt_reader = imageio.get_reader(str(gt_input_path))
    camera_readers = [imageio.get_reader(str(path)) for path in camera_paths]
    alloc_readers = [imageio.get_reader(str(path)) for path in alloc_paths]
    
    readers = [gt_reader, *camera_readers, *alloc_readers]
    
    try:
        # Determine square tile size from GT input (top row)
        gt_meta = gt_reader.get_meta_data()
        top_height = gt_meta["size"][1]
        top_width = gt_meta["size"][0]
        tile_size = min(top_height, top_width)
        
        output_height = tile_size * 2
        output_width = tile_size * 4
        
        LOGGER.info(f"Tile size: {tile_size}, output: {output_width}x{output_height}")
        
        # Frame count limited by all videos
        num_frames = min(
            gt_reader.count_frames(),
            min(r.count_frames() for r in camera_readers),
            min(r.count_frames() for r in alloc_readers),
        )
        
        LOGGER.info(f"Merging {num_frames} frames...")
        
        output_writer = imageio.get_writer(
            str(output_path),
            fps=video_fps,
            codec="libx264",
            quality=8,
        )
        
        def crop_square(arr: np.ndarray) -> np.ndarray:
            h, w = arr.shape[:2]
            size = min(h, w)
            top = (h - size) // 2
            left = (w - size) // 2
            return arr[top:top + size, left:left + size]
        
        try:
            for frame_idx in tqdm(range(num_frames), desc="Merging aligned frames"):
                # Top row frames
                gt_frame = gt_reader.get_data(frame_idx)
                camera_frames = [reader.get_data(frame_idx) for reader in camera_readers]
                
                top_frames = [gt_frame, *camera_frames]
                top_squares = []
                for frame in top_frames:
                    square = crop_square(frame) if frame.shape[0] != frame.shape[1] else frame
                    if square.shape[0] != tile_size:
                        square = np.array(
                            Image.fromarray(square).resize(
                                (tile_size, tile_size), Image.Resampling.LANCZOS
                            )
                        )
                    top_squares.append(square[:, :, :3])
                
                # Bottom row frames
                alloc_frames = [reader.get_data(frame_idx) for reader in alloc_readers]
                bottom_squares = []
                for frame in alloc_frames:
                    square = crop_square(frame)
                    if square.shape[0] != tile_size:
                        square = np.array(
                            Image.fromarray(square).resize(
                                (tile_size, tile_size), Image.Resampling.LANCZOS
                            )
                        )
                    bottom_squares.append(square[:, :, :3])
                
                composite = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                
                # Place top row
                for idx, square in enumerate(top_squares):
                    x0 = idx * tile_size
                    composite[:tile_size, x0:x0 + tile_size] = square
                
                # Place bottom row
                for idx, square in enumerate(bottom_squares):
                    x0 = idx * tile_size
                    composite[tile_size:, x0:x0 + tile_size] = square
                
                output_writer.append_data(composite)
        finally:
            output_writer.close()
        
        LOGGER.info(f"Created aligned comparison video: {output_path}")
    finally:
        for reader in readers:
            reader.close()

def merge_videos_cmp_hoi(
    seq_obj: str,
    alloc_path: Path,
    method_list: Sequence[str],
    image_folder: str = "images",
    video_fps: int = 30,
) -> None:
    """Merge videos into a comparison layout for HOI.
    
    Layout:
    - Top row: gt_input.mp4 | fp_simple_camera.mp4 | fp_full_camera.mp4 | ours_camera.mp4
    - Bottom row: {seq_obj}_joint_alloc.mp4 (full width)
    
    Args:
        seq_obj: Sequence/object identifier
        alloc_path: Path to directory containing joint alloc videos
        method_list: List of method names for camera videos
        image_folder: Image folder name (default: "images")
        video_fps: Video FPS (default: 30)
    """
    _ensure_logger()
    
    # Paths to videos
    gt_image_dir = SAVE_ROOT / "gt" / seq_obj / image_folder
    gt_video_camera_dir = gt_image_dir / "video_camera_frames"
    gt_input_path = gt_image_dir / f"{seq_obj}_gt_input.mp4"
    
    camera_paths = []
    for method in method_list:
        camera_path = SAVE_ROOT / method / seq_obj / image_folder / f"{seq_obj}_{method}_camera.mp4"
        camera_paths.append(camera_path)
    
    joint_alloc_path = alloc_path / f"{seq_obj}_joint_alloc.mp4"
    output_path = SAVE_ROOT / "ours" / "cmp_hoi" / f"{seq_obj}_cmp_hoi.mp4"
    if output_path.exists():
        LOGGER.info(f"Comparison video already exists, skipping: {output_path}")
        return
    
    # Create GT input video from frames if it doesn't exist
    if not gt_input_path.exists():
        LOGGER.info(f"Creating GT input video from frames in: {gt_video_camera_dir}")
        if not gt_video_camera_dir.exists():
            LOGGER.error(f"GT video camera frames directory not found: {gt_video_camera_dir}")
            return
        
        # Check if input frames exist
        input_frame_paths = sorted(gt_video_camera_dir.glob("*_input.png"))
        if not input_frame_paths:
            LOGGER.error(f"No input frames found matching '*_input.png' in {gt_video_camera_dir}")
            return
        
        LOGGER.info(f"Found {len(input_frame_paths)} GT input frames")
        
        # Create GT input video from frames
        _create_video_from_frames(
            gt_video_camera_dir,
            gt_input_path,
            fps=video_fps,
            pattern="*_input.png"
        )
        
        # Verify GT input video was created
        if not gt_input_path.exists():
            LOGGER.error(f"Failed to create GT input video: {gt_input_path}")
            return
        
        LOGGER.info(f"GT input video created: {gt_input_path}")
    
    # Check if all videos exist
    if not gt_input_path.exists():
        LOGGER.warning(f"GT input video not found: {gt_input_path}")
        return
    
    for method, camera_path in zip(method_list, camera_paths):
        if not camera_path.exists():
            LOGGER.warning(f"Camera video not found for {method}: {camera_path}")
            return
    
    if not joint_alloc_path.exists():
        LOGGER.warning(f"Joint alloc video not found: {joint_alloc_path}")
        return
    
    LOGGER.info(f"Merging comparison videos for {seq_obj}")
    
    # Read all videos
    readers = []

    gt_reader = imageio.get_reader(str(gt_input_path))
    readers.append(gt_reader)
    
    camera_readers = []
    for camera_path in camera_paths:
        reader = imageio.get_reader(str(camera_path))
        camera_readers.append(reader)
        readers.append(reader)
    
    alloc_reader = imageio.get_reader(str(joint_alloc_path))
    readers.append(alloc_reader)
    
    # Get dimensions from joint alloc video (this will be the bottom row, full width)
    alloc_meta = alloc_reader.get_meta_data()
    alloc_height = alloc_meta['size'][1]
    alloc_width = alloc_meta['size'][0]
    
    # Top row videos should be smaller - use alloc_width / 4 for each (4 videos in top row)
    top_video_width = alloc_width // 4
    top_video_height = top_video_width
    
    # Total output dimensions: (alloc_height + top_video_height, alloc_width)
    output_height = alloc_height + top_video_height
    output_width = alloc_width
    
    LOGGER.info(f"Output dimensions: {output_width}x{output_height}")
    LOGGER.info(f"Top row videos: {top_video_width}x{top_video_height} each")
    LOGGER.info(f"Bottom row (alloc): {alloc_width}x{alloc_height}")
    
    # Get frame counts
    num_frames = min(
        gt_reader.count_frames(),
        min(r.count_frames() for r in camera_readers),
        alloc_reader.count_frames()
    )
    
    LOGGER.info(f"Merging {num_frames} frames...")
    
    # Create output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_writer = imageio.get_writer(
        str(output_path),
        fps=video_fps,
        codec="libx264",
        quality=8
    )
        

    for frame_idx in tqdm(range(num_frames), desc="Merging comparison frames"):
        # Read frames
        gt_frame = gt_reader.get_data(frame_idx)
        camera_frames = [reader.get_data(frame_idx) for reader in camera_readers]
        alloc_frame = alloc_reader.get_data(frame_idx)
        
        # Resize top row videos
        gt_img = Image.fromarray(gt_frame).resize((top_video_width, top_video_height), Image.Resampling.LANCZOS)
        camera_imgs = [
            Image.fromarray(cam_frame).resize((top_video_width, top_video_height), Image.Resampling.LANCZOS)
            for cam_frame in camera_frames
        ]
        
        # Convert to numpy
        gt_arr = np.array(gt_img)[:, :, :3]
        camera_arrs = [np.array(img)[:, :, :3] for img in camera_imgs]
        alloc_arr = alloc_frame[:, :, :3]
        
        # Create composite frame
        composite = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Top row: gt_input | fp_simple_camera | fp_full_camera | ours_camera
        x_offset = 0
        composite[:top_video_height, x_offset:x_offset+top_video_width] = gt_arr
        x_offset += top_video_width
        for camera_arr in camera_arrs:
            composite[:top_video_height, x_offset:x_offset+top_video_width] = camera_arr
            x_offset += top_video_width
        
        # Bottom row: joint alloc (full width)
        composite[top_video_height:, :] = alloc_arr
        
        # Write frame
        output_writer.append_data(composite)
    

    output_writer.close()
        
    LOGGER.info(f"Created comparison video: {output_path}")
    
    for reader in readers:
        reader.close()


def merge_videos_for_seq_obj_list(
    seq_obj_list: Sequence[str],
    method: str,
    image_folder: str = "images",
    video_fps: int = 30,
    mode: str = "3way",
) -> None:
    """Merge videos for a list of sequence/object pairs.
    
    Args:
        seq_obj_list: List of sequence/object identifiers
        method: Method name
        image_folder: Image folder name (default: "images")
        video_fps: Video FPS (default: 30)
    """
    _ensure_logger()
    
    for seq_obj in tqdm(seq_obj_list, desc="Merging videos"):
        if mode == "3way":
            merge_videos_3way(seq_obj, method, image_folder, video_fps)
        elif mode == "cmp_hoi":
            alloc_path = SAVE_ROOT / f"cmp_gt+fp_simple+fp_full+ours/{seq_obj}/video_new/"
            method_list = ["fp_simple", "fp_full", "ours"]
            merge_videos_cmp_hoi(seq_obj, alloc_path, method_list, image_folder, video_fps)
        elif mode == "cmp_hoi_aligned":
            method_list = ["fp_simple", "fp_full", "ours"]
            merge_videos_cmp_hoi_aligned(seq_obj, SAVE_ROOT, method_list, image_folder, video_fps)




def merge_to_ours(
    seq_obj: str | None = None,
    method: str = "ours",
    image_folder: str = "video",
    video_fps: int = 30,
    split: str = "test50obj",
    mode: str = "3way",
) -> None:
    """Main function to merge videos for one or more sequence/object pairs.
    
    Args:
        seq_obj: Single sequence/object identifier, or None to use split
        method: Method name (default: "ours")
        image_folder: Image folder name (default: "images")
        video_fps: Video FPS (default: 30)
        split: Split name to use if seq_obj is None (default: "test50obj")
    """
    _ensure_logger()
    
    if seq_obj is None:
        import json
        import os.path as osp
        split_file = osp.join("data/HOT3D-CLIP", "sets", "split.json")
        with open(split_file, "r", encoding="utf-8") as f:
            split_dict = json.load(f)
        seq_obj_list = split_dict[split]
    else:
        seq_obj_list = [seq_obj]
    
    merge_videos_for_seq_obj_list(seq_obj_list, method, image_folder, video_fps, mode=mode)


if __name__ == "__main__":
    from fire import Fire
    Fire(merge_to_ours)
