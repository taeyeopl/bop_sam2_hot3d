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
import cv2

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import datetime

# HT toolkit (https://github.com/facebookresearch/hand_tracking_toolkit)
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection, warp_image
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from hot3d_clips_utils import clip_util

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from PIL import Image


# checkpoint = "../segment_anything_v2/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
checkpoint = "../segment_anything_v2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)


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
        conf_thres=90,
        iou_thres=75,
        target_object=None
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

    total_frame = (clip_util.get_number_of_frames)(tar)

    # Per-frame visualization.
    frame_id = 0
    frame_key = f"{frame_id:06d}"

    # Load camera parameters.
    cameras = clip_util.load_cameras(tar, frame_key)

    # Available image streams.
    image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

    start_time = datetime.datetime.now()
    print(f'\n\nTime: {start_time.strftime("%Y-%m-%d %H:%M:%S")} start: {clip_output_path}\n\n')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for stream_id in image_streams:

        rgb_path = os.path.join(os.path.join(clip_output_path, stream_id), 'rgb')
        mask_path = os.path.join(os.path.join(clip_output_path, stream_id), 'mask')
        sam_path = os.path.join(os.path.join(clip_output_path, stream_id), 'sam')
        mask_visib_path = os.path.join(os.path.join(clip_output_path, stream_id), 'mask_visib')

        vis_sam_output_path = os.path.join(os.path.join(clip_output_path, stream_id), 'sam_result.npy')

        sam_conf_visibility_all = np.load(vis_sam_output_path)

        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(sam_path, exist_ok=True)
        os.makedirs(mask_visib_path, exist_ok=True)

        # Load hand and object annotations.
        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

        inference_state = predictor.init_state(video_path=rgb_path)

        # Visualize object contours.
        for obj_idx, instance_list in tqdm.tqdm(enumerate(objects.values()), desc='process object'):
            if target_object != None:
                if obj_idx != target_object :
                    continue
            predictor.reset_state(inference_state)

            tmp_visibility_set = []
            sam_mask_size = []
            not_on_edge_check = []

            for i in range(total_frame):
                vis_sam_path = os.path.join(sam_path, f'{i:06d}_{obj_idx:06d}.png')
                vis_mask_path = os.path.join(mask_path, f'{i:06d}_{obj_idx:06d}.png')
                sam_mask = imageio.v3.imread(vis_sam_path)
                render_amodal_mask = imageio.v3.imread(vis_mask_path)
                dilated_render_amodal_mask = cv2.dilate(render_amodal_mask, kernel, iterations=1)
                tmp_visibility = clip_util.calculate_visibility_iou(sam_mask.astype(np.bool), dilated_render_amodal_mask.astype(np.bool))

                tmp_visibility_set.append(tmp_visibility)
                sam_mask_size.append(sam_mask.astype(np.bool).sum())

                not_on_edge = clip_util.check_bbox_not_on_edge(render_amodal_mask)
                not_on_edge_check.append(not_on_edge)

            tmp_visibility_set = np.array(tmp_visibility_set)
            sam_mask_size = np.array(sam_mask_size)
            not_on_edge_check = np.array(not_on_edge_check)

            # choose the frame_id
            sam_conf_visibility = sam_conf_visibility_all[obj_idx][...,:2]
            sam_conf_visibility[..., 1] = tmp_visibility_set
            sam_conf = sam_conf_visibility[..., 0]
            sam_vis = sam_conf_visibility[..., 1]
            conf = sam_conf + sam_vis

            top_1_indices_per_step = []

            max_index = clip_util.get_max_index(not_on_edge_check, conf, sam_conf, sam_vis, conf_thres, iou_thres, sam_mask_size)

            if debug:
                mask_visib_test_path = os.path.join(os.path.join(clip_output_path, stream_id), f'vis_{conf_thres}_iou_{iou_thres}/obj_{obj_idx:06d}_max_{max_index}')
                os.makedirs(mask_visib_test_path, exist_ok=True)


            if (sam_conf[max_index] >= conf_thres) and (sam_vis[max_index] >= iou_thres):
                print(f'sam_conf[{max_index}] ({sam_conf[max_index]}), sam_vis[{max_index}] ({sam_vis[max_index]})' )
                top_1_indices_per_step.append(max_index)


            video_segments = {}  # video_segments contains the per-frame segmentation results
            if len(top_1_indices_per_step) == 0:
                pass
            else:
                # Add new masks for SAM2
                for index in top_1_indices_per_step:
                    vis_sam_path = os.path.join(sam_path, f'{index:06d}_{obj_idx:06d}.png')
                    s_mask = imageio.v2.imread(vis_sam_path)

                    frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=index,
                        obj_id=255,
                        mask=s_mask,
                        )

                # Forward
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    for i, out_obj_id in enumerate(out_obj_ids):
                        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()}

                # Estimate total full frames by forward pass
                if len(video_segments) == total_frame:
                    pass
                else:
                    # Backward
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                        # Update Backward When Empty
                        if not (out_frame_idx in video_segments):
                            for i, out_obj_id in enumerate(out_obj_ids):
                                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()}

            clip_util.process_video(debug, total_frame, obj_idx, video_segments, rgb_path, mask_visib_path, mask_visib_test_path if debug else None, clip_output_path, stream_id, clip_name, sam_conf_visibility, max_index, conf_thres, iou_thres)

    current_time = datetime.datetime.now()
    time_difference = current_time - start_time

    print(f'\n\nTime: {current_time.strftime("%Y-%m-%d %H:%M:%S")} Finish: {clip_output_path}\n\n')
    print(f'\n\nTime taken: {time_difference}\n\n')


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
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            target_object=args.target_object
            )


if __name__ == "__main__":
    main()
