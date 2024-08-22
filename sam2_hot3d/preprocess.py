#!/usr/bin/env python3

import argparse
import os
import tarfile
import imageio
import sys
import numpy as np
import trimesh
import pickle
import torch
import tqdm
import json
import datetime

from typing import Any, Dict, List, Optional

# HT toolkit (https://github.com/facebookresearch/hand_tracking_toolkit)
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection, warp_image
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from hot3d_clips_utils import clip_util

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "../segment_anything_v2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
# checkpoint = "../segment_anything_v2/checkpoints/sam2_hiera_tiny.pt"
# model_cfg = "sam2_hiera_t.yaml"

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


@torch.inference_mode()
@torch.cuda.amp.autocast()
def vis_clip(
        clip_path: str,
        object_models: Dict[int, trimesh.Trimesh],
        hand_type: str,
        mano_model: Optional[MANOHandModel],
        undistort: bool,
        output_dir: str,
        frame_start=0,
        frame_end=-1,
        debug=False,
        ) -> None:
    """Visualizes hand and object models in GT poses for each frame of a clip.

    Args:
        clip_path: Path to a clip saved as a tar file.
        object_models: Mapping from BOP object ID to the object mesh.
        hand_type: Hand type ("umetrack" or "mano").
        mano_model: MANO hand model (needs to be provided
            if hand_type == "mano").
        undistort: Whether to undistort the fisheye images.
        output_dir: Path to a folder for output visualizations.
    """

    # Open the tar file with clip data.
    tar = tarfile.open(clip_path, mode="r")

    # Prepare output folder.
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(output_dir, clip_name)

    os.makedirs(clip_output_path, exist_ok=True)

    total_frame = 5 if debug else (clip_util.get_number_of_frames)(tar)

    # Per-frame visualization.
    clip_214_1 = []
    clip_1201_1 = []
    clip_1201_2 = []

    start_time = datetime.datetime.now()
    print(f'Time: {start_time.strftime("%Y-%m-%d %H:%M:%S")} start: {clip_output_path}')

    for frame_id, count in enumerate(tqdm.tqdm(range(total_frame),desc='processing frame')):

        # Skip the clip if it is not in the specified range.
        if count < frame_start or (frame_end >= 0 and count > frame_end):
            continue

        frame_key = f"{frame_id:06d}"

        # Load camera parameters.
        cameras = clip_util.load_cameras(tar, frame_key)

        # Available image streams.
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        for stream_id in image_streams:

            rgb_path = os.path.join(os.path.join(clip_output_path, stream_id), 'rgb')
            mask_path = os.path.join(os.path.join(clip_output_path, stream_id), 'mask')
            sam_path = os.path.join(os.path.join(clip_output_path, stream_id), 'sam')

            os.makedirs(rgb_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)
            os.makedirs(sam_path, exist_ok=True)

            # Load hand and object annotations.
            objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

            stream_key = str(stream_id)

            # Load the image.
            image: np.ndarray = clip_util.load_image(tar, frame_key, stream_key)

            # Make sure the image has 3 channels.
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            # Write the png  RGB for SAM2
            vis_rgb_path = os.path.join(rgb_path, f'{frame_id:06d}.png')
            imageio.imwrite(vis_rgb_path, image)

            # Camera parameters of the current image.
            camera_model = cameras[stream_id]

            render_masks = []

            # Visualize object contours.
            for obj_idx, instance_list in enumerate(objects.values()):
                for instance in instance_list:
                    bop_id = int(instance["object_bop_id"])

                    # Transformation from the model to the world space.
                    T = clip_util.se3_from_dict(instance["T_world_from_object"])

                    # Vertices in the model space.
                    verts_in_m = object_models[bop_id].vertices

                    verts_in_w = (T[:3, :3] @ verts_in_m.T + T[:3, 3:]).T

                    # Render the object model (outputs: rgb, mask, depth).
                    ren_obj_rgb, ren_obj_mask, ren_obj_depth = rasterizer.rasterize_mesh(
                        verts=verts_in_w,
                        faces=object_models[bop_id].faces,
                        vert_normals=object_models[bop_id].vertex_normals,
                        camera=camera_model,
                        )

                    ren_obj_mask[ren_obj_mask != 0] = 1
                    ren_obj_mask = ren_obj_mask * 255
                    render_masks.append(ren_obj_mask)

            render_masks = np.array(render_masks)

            # run sam2
            bboxes = clip_util.get_bounding_boxes_batch(render_masks, pad_rel=0.0)
            predictor.set_image(image)
            sam_masks, sam_iou_predictions, sam_low_res_masks = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
                )
            sam_masks = sam_masks.squeeze(1)
            sam_iou_predictions = sam_iou_predictions.squeeze(1)

            # save render, sam_mask, confidence
            for idx, (s_mask, s_conf, r_mask) in enumerate(zip(sam_masks, sam_iou_predictions, render_masks)):
                vis_mask_path = os.path.join(mask_path, f'{frame_id:06d}_{idx:06d}.png')
                vis_sam_path = os.path.join(sam_path, f'{frame_id:06d}_{idx:06d}.png')
                imageio.imwrite(vis_mask_path, r_mask)

                s_mask = (s_mask * 255).astype(np.uint8)
                imageio.imwrite(vis_sam_path, s_mask)

                # save sam confidence and visibility iou
                s_conf = np.round(s_conf * 100.0).astype(np.uint8)

                not_on_edge = clip_util.check_bbox_not_on_edge(r_mask)
                visibility = clip_util.calculate_visibility_iou(s_mask.astype(np.bool), r_mask.astype(np.bool))



                data = {
                    'conf': int(s_conf),
                    'visibility': int(visibility),
                    'size': int(s_mask.sum()),
                    'idx': int(idx),
                    'not_on_edge': not_on_edge,
                    'frame_id': int(frame_id),
                    }

                if stream_id == '214-1':
                    clip_214_1.append(data)
                elif stream_id == '1201-1':
                    clip_1201_1.append(data)
                elif stream_id == '1201-2':
                    clip_1201_2.append(data)

    # save sam confidence and iou visibility
    for stream_id in image_streams:
        if stream_id == '214-1':
            input_array = np.array(clip_214_1)
        elif stream_id == '1201-1':
            input_array = np.array(clip_1201_1)
        elif stream_id == '1201-2':
            input_array = np.array(clip_1201_2)
        max_idx = max(item['idx'] for item in input_array)
        max_frame_id = max(item['frame_id'] for item in input_array)

        # Initialize a 3D array to store the results
        result = np.zeros((max_idx + 1, max_frame_id + 1, 4), dtype=float)

        vis_sam_output_path = os.path.join(os.path.join(clip_output_path, stream_id), 'sam_result.npy')

        # Reconstruct the data
        for item in input_array:
            idx = item['idx']
            frame_id = item['frame_id']
            result[idx][frame_id][0] = item['conf']
            result[idx][frame_id][1] = item['visibility']
            result[idx][frame_id][2] = item['size']
            result[idx][frame_id][3] = item['not_on_edge']

        np.save(vis_sam_output_path, result)


    end_time = datetime.datetime.now()
    time_difference = end_time - start_time

    print(f'Time: {end_time.strftime("%Y-%m-%d %H:%M:%S")} Finish: {clip_output_path}')
    print(f'Time taken: {time_difference}')
def main() -> None:

    args = clip_util.parse_arguments()

    # Make sure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load object models.
    save_path = '../object_models.pkl'

    if os.path.exists(save_path):
        object_models = clip_util.load_models(save_path)
    else:
        object_models: Dict[int, trimesh.Trimesh] = {}
        object_model_filenames = sorted(
            [p for p in os.listdir(args.object_models_dir) if p.endswith(".glb")]
            )
        for model_filename in object_model_filenames:
            model_path = os.path.join(args.object_models_dir, model_filename)
            print(f"Loading model: {model_path}")
            object_id = int(model_filename.split(".glb")[0].split("obj_")[1])
            object_models[object_id] = clip_util.load_mesh(model_path)

        # save object models for saving time
        with open(save_path, 'wb') as f:
            pickle.dump(object_models, f)
        print(f"Models saved to {save_path}")

    # Optionally load MANO hand model.
    mano_model = None
    if args.hand_type == "mano":
        mano_model = MANOHandModel(args.mano_model_dir)

    # List clips present in the source folder.
    clips = sorted([p for p in os.listdir(args.clips_dir) if p.endswith(".tar")])

    # Visualize the clips.
    for clip in tqdm.tqdm(clips, desc='processing clip'):
        clip_id = int(clip.split(".tar")[0].split("clip-")[1])

        # Skip the clip if it is not in the specified range.
        if clip_id < args.clip_start or (args.clip_end >= 0 and clip_id > args.clip_end):
            continue

        vis_clip(
            clip_path=os.path.join(args.clips_dir, clip),
            object_models=object_models,
            hand_type=args.hand_type,
            mano_model=mano_model,
            undistort=args.undistort,
            output_dir=args.output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            debug=args.debug,
            )


if __name__ == "__main__":
    main()
