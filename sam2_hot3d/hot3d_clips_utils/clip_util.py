#!/usr/bin/env python3

import json
import cv2
import imageio
import numpy as np
import torch
import trimesh
from typing import Any, Dict, Optional, Sequence, Tuple

# HT toolkit (https://github.com/facebookresearch/hand_tracking_toolkit)
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import (
    decode_hand_pose, HandShapeCollection, HandSide
    )
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    from_json as from_umetrack_hand_model_json,
    )
from hand_tracking_toolkit import math_utils, visualization

import matplotlib

# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
from sklearn import linear_model
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import PolynomialFeatures

import argparse

import pickle
from torchvision.transforms.functional import to_tensor

import matplotlib.patches as patches
import os
from PIL import Image
import subprocess
import glob


def check_bbox_not_on_edge(render_amodal_mask, margin=5):
    height, width = render_amodal_mask.shape
    y_indices, x_indices = np.where(render_amodal_mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        bbox = np.array([0, 0, 0, 0])
    else:
        # Calculate the coordinates of the bounding box
        left = np.min(x_indices)
        top = np.min(y_indices)
        right = np.max(x_indices)
        bottom = np.max(y_indices)
        bbox = np.array([left, top, right, bottom])

    # Check if the bounding box is on the edge of the image
    not_on_edge = (
            bbox[0] >= margin and  # Left edge
            bbox[1] >= margin and  # Top edge
            bbox[2] < width - margin and  # Right edge
            bbox[3] < height - margin  # Bottom edge
    )

    return not_on_edge


def get_max_index(edge_check, conf, sam_conf, sam_vis, conf_thres, iou_thres, sam_mask_size):
    # First, filter indices based on edge check
    valid_edge_indices = np.where(edge_check)[0]

    if len(valid_edge_indices) == 0:
        # If no valid edges, fall back to the original logic
        return np.argmax(conf)

    # Filter confidence scores for valid edge indices
    valid_edge_conf = conf[valid_edge_indices]

    # Find max confidence among valid edge indices
    max_conf = np.max(valid_edge_conf)
    max_indices = np.where(valid_edge_conf == max_conf)[0]

    if len(max_indices) == 1:
        return valid_edge_indices[max_indices[0]]

    # Apply additional filters on max confidence indices
    valid_conf = sam_conf[valid_edge_indices[max_indices]] >= conf_thres
    valid_iou = sam_vis[valid_edge_indices[max_indices]] >= iou_thres

    common_indices = np.where(valid_conf & valid_iou)[0]

    if len(common_indices) > 0:
        # Select the index with the largest mask size among common indices
        largest_mask_index = np.argmax(sam_mask_size[valid_edge_indices[max_indices[common_indices]]])
        return valid_edge_indices[max_indices[common_indices[largest_mask_index]]]

    # If no indices meet all criteria, return the index with max confidence among valid edge indices
    return valid_edge_indices[max_indices[0]]

def create_gif_mp4(obj_idx, mask_visib_test_path, clip_output_path, stream_id, max_index, conf_thres, iou_thres, use_mp4=True, use_gif=True):
    # Get sorted list of PNG files
    png_pattern = os.path.join(mask_visib_test_path, f'*_{obj_idx:06d}.png')
    png_files = sorted(glob.glob(png_pattern))

    if not png_files:
        print(f"No PNG files found matching the pattern: {png_pattern}")
        return None, None

    output_dir = os.path.join(clip_output_path, stream_id)
    os.makedirs(output_dir, exist_ok=True)

    base_name = f'vis_{obj_idx:06d}_conf_{conf_thres}_iou_{iou_thres}_max_index_{max_index}'

    if use_gif:
        gif_path = os.path.join(output_dir, f'{base_name}.gif')
        # Create GIF
        gif_command = f"convert {' '.join(png_files)} -resize 480x {gif_path}"
        gif_result = subprocess.run(gif_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if gif_result.returncode != 0:
            print(f"Error creating GIF: {gif_result.stderr.decode()}")
        else:
            print(f"GIF created successfully: {gif_path}")

    if use_mp4:
        # Create MP4
        mp4_path = os.path.join(output_dir, f'{base_name}.mp4')
        # mp4_command = f"ffmpeg -framerate 10 -i {mask_visib_test_path}/%06d_{obj_idx:06d}.png {mp4_path}"
        mp4_command = (
            f"ffmpeg -framerate 10 -pattern_type glob -i '{png_pattern}' "
            f"-vf 'scale=480:-1' -c:v libx264 -pix_fmt yuv420p {mp4_path}"
        )

        mp4_result = subprocess.run(mp4_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if mp4_result.returncode != 0:
            print(f"Error creating MP4: {mp4_result.stderr.decode()}")
        else:
            print(f"MP4 created successfully: {mp4_path}")


def process_frame(clip_name, frame_idx, obj_idx, video_segments, rgb_path, mask_visib_path, mask_visib_test_path=None, sam_conf_visibility=None, max_index=0, conf_thres=0, iou_thres=0):
    final_mask_path = os.path.join(mask_visib_path, f'{frame_idx:06d}_{obj_idx:06d}.png')
    rendered_amodal_mask_path = final_mask_path.replace(f'mask_visib','mask')
    rendered_amodal_mask = imageio.v2.imread(rendered_amodal_mask_path).astype(np.uint8)
    iou = 0

    if len(video_segments)==0:
        final_mask = np.zeros(rendered_amodal_mask.shape).astype(np.uint8)
        iou = 0
    else:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            final_mask = (out_mask[0] * 255).astype(np.uint8)
            final_mask[rendered_amodal_mask==False] = 0
            iou = calculate_visibility_iou(final_mask, rendered_amodal_mask)

    # save final_mask
    imageio.imwrite(final_mask_path, final_mask)

    if mask_visib_test_path:
        sam_conf, sam_vis = sam_conf_visibility[frame_idx]
        clip_index = int(clip_name.split('-')[-1])

        vis_path = os.path.join(mask_visib_test_path, f'{frame_idx:06d}_{obj_idx:06d}.png')
        plt.figure(figsize=(3, 2))
        plt.title(f"vis({iou}) / max({int(max_index)}) conf({int(sam_conf)}/{int(conf_thres)}) iou({int(sam_vis)}/{int(iou_thres)}) / f({frame_idx}) c({clip_index}) o({obj_idx})", fontsize=4)
        plt.imshow(Image.open(os.path.join(rgb_path, f'{frame_idx:06d}.png')))

        show_mask(rendered_amodal_mask.astype(bool), plt.gca(), obj_id=255)
        show_mask(final_mask.astype(bool), plt.gca(), obj_id=1)

        plt.axis('off')
        plt.savefig(vis_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close("all")

    return iou
def process_video(debug, total_frame, obj_idx, video_segments, rgb_path, mask_visib_path, mask_visib_test_path, clip_output_path, stream_id, clip_name, sam_conf_visibility, max_index, conf_thres, iou_thres):
    visibility = []

    vis_frame_stride = 1 #30 if debug else 1

    if debug:
        plt.close("all")
    for frame_idx in range(0, total_frame, vis_frame_stride):
        iou = process_frame(clip_name, frame_idx, obj_idx, video_segments, rgb_path, mask_visib_path, mask_visib_test_path if debug else None, sam_conf_visibility, max_index, conf_thres, iou_thres)
        visibility.append(iou)

    # save visibility
    visibility_path = os.path.join(os.path.join(clip_output_path, stream_id), f'obj_{obj_idx:06d}_visibility.npy')
    np.save(visibility_path, visibility)

    if debug:
        create_gif_mp4(obj_idx, mask_visib_test_path, clip_output_path, stream_id, max_index, conf_thres, iou_thres)

    return visibility

    # calculate visibility



def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def create_visibility_plot(clip_render_mask, clip_image, clip_sam_mask, est_mask, f_idx, o_idx, clip_id, clip_output_path, clip_name, stream_id, clip_visibility_max_index, visibility,init=False, start_index=0):
    ren_mask = clip_render_mask[f_idx][o_idx]
    visibility_iou = calculate_visibility_iou(est_mask, ren_mask)
    visibility_sam_iou = calculate_visibility_iou(clip_sam_mask[f_idx][o_idx], ren_mask)

    iou_sam_est_mask = calculate_iou(est_mask, clip_sam_mask[f_idx][o_idx])
    iou_est_ren_mask = calculate_iou(est_mask, ren_mask)
    ren_sam_mask = calculate_iou(ren_mask, clip_sam_mask[f_idx][o_idx])

    est_mask_size = est_mask.sum()
    sam_mask_size = clip_sam_mask[f_idx][o_idx].sum()

    # est_confidence = np.round(est_mask_size.astype(np.float) * visibility_iou.astype(np.float) / 100.0 ).astype(np.uint16)
    # sam_confidence = np.round(sam_mask_size.astype(np.float) * visibility_sam_iou.astype(np.float) / 100.0).astype(np.uint16)


    d_id = clip_id[f_idx][o_idx]

    ph, pw = 2, 3
    if start_index == 999:
        vis_obj_path = os.path.join(clip_output_path, f'obj_{d_id:02d}_combine')
        os.makedirs(vis_obj_path, exist_ok=True)
        vis_dir = f"{clip_name}_{f_idx:03d}_{stream_id}_obj_{d_id:02d}_combine"
    else:
        vis_obj_path = os.path.join(clip_output_path, f'obj_{d_id:02d}_start_{start_index:03d}')
        os.makedirs(vis_obj_path, exist_ok=True)
        vis_dir = f"{clip_name}_{f_idx:03d}_{stream_id}_obj_{d_id:02d}_start_{start_index:03d}"
    vis_path = os.path.join(vis_obj_path, vis_dir)

    fig, axes = plt.subplots(ph, pw, figsize=(6, 5))
    axes = axes.ravel()  # Flatten the 2D array of axes for easier iteration

    row_min, row_max, col_min, col_max = calculate_expanded_box(ren_mask)

    # First subplot
    axes[0].imshow(clip_image[f_idx])
    axes[0].axis('off')
    axes[0].set_title('image', fontsize=8)
    axes[0].text(0.5, 0.95, vis_dir, ha='center', va='top', transform=axes[0].transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=5)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    tmp = clip_image[f_idx].copy()
    tmp[est_mask == 0] = [0, 255, 0]
    axes[1].imshow(tmp)
    axes[1].axis('off')
    axes[1].set_title('image (est_mask)', fontsize=8)
    axes[1].text(0.5, 0.95, f'est_vis ({visibility_iou}%) est_size ({est_mask_size}px) \n'
                            f' iou_est_ren ({iou_est_ren_mask}%) iou_est_sam ({iou_sam_est_mask}%)', ha='center', va='top', transform=axes[1].transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=5)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    axes[1].add_patch(rect)

    # Second subplot
    tmp = clip_image[f_idx].copy()
    tmp[ren_mask == 0] = [0, 255, 0]
    axes[2].imshow(tmp)
    axes[2].axis('off')
    axes[2].set_title('image (ren_mask)', fontsize=8)
    axes[2].text(0.5, 0.95, f'Start Frame ({start_index})', ha='center', va='top',
                 transform=axes[2].transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=5)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    axes[2].add_patch(rect)

    axes[3].imshow(clip_sam_mask[f_idx][o_idx])
    axes[3].axis('off')
    axes[3].set_title('sam_mask', fontsize=8)
    sam_net_conf = np.round(visibility[3][f_idx][o_idx]*100.0).astype(np.uint8)
    axes[3].text(0.5, 0.95, f'sam_conf ({sam_net_conf}%) sam_size ({sam_mask_size}px) \n '
                            f'iou_sam_ren ({ren_sam_mask}%) iou_sam_est ({iou_sam_est_mask}%)'
                            f'', ha='center', va='top', transform=axes[3].transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=5)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    axes[3].add_patch(rect)


    axes[4].imshow(est_mask)
    axes[4].axis('off')
    axes[4].set_title('est_mask', fontsize=8)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    if init:
        axes[4].text(0.5, 0.95, 'init', ha='center', va='top', transform=axes[4].transAxes,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=5)
    axes[4].add_patch(rect)

    axes[5].imshow(ren_mask)
    axes[5].axis('off')
    axes[5].set_title('ren_mask', fontsize=8)
    rect = patches.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min, linewidth=1,
                             edgecolor='r', facecolor='none')
    axes[5].add_patch(rect)

    plt.savefig(vis_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.close()

    return iou_sam_est_mask, iou_est_ren_mask, ren_sam_mask

def calculate_expanded_box(d_model, box_ratio=0.7):
    """
    Calculate an expanded bounding box based on non-zero elements in a tensor.

    Parameters:
        d_model (torch.Tensor): A 2D tensor where non-zero elements define the bounding box.
        box_ratio (float): The ratio by which to expand the bounding box.

    Returns:
        tuple: Expanded bounding box (row_min, row_max, col_min, col_max)
    """
    # Find non-zero elements
    rows, cols = d_model.nonzero()

    if len(rows) == 0 or len(cols) == 0:
        row_min, row_max = 0,0
        col_min, col_max = 0,0
    else:
        row_min, row_max = rows.min().item(), rows.max().item()
        col_min, col_max = cols.min().item(), cols.max().item()

    # Calculate height and width of the bounding box
    box_height = row_max - row_min
    box_width = col_max - col_min

    # Expand the bounding box by the given ratio
    row_min = int(row_min - box_ratio * box_height)
    row_max = int(row_max + box_ratio * box_height)
    col_min = int(col_min - box_ratio * box_width)
    col_max = int(col_max + box_ratio * box_width)

    return row_min, row_max, col_min, col_max


def process_image_mask(image, mask, clip_render_mask, processor, f_idx, o_idx, init=False):
    """
    Process an image and its corresponding mask to generate output masks.

    Parameters:
        image (np.ndarray): The image to be processed.
        mask (np.ndarray): The mask corresponding to the image.
        processor (object): The processor object that has `step` and `output_prob_to_mask` methods.
        f_idx (int): The index for selecting the image and mask.
        o_idx (int): The index for selecting the mask object.
        clip_render_mask (np.ndarray): The clip render mask to refine the output mask.

    Returns:
        np.ndarray: The processed mask as a NumPy array.
    """
    image_tensor = to_tensor(image[f_idx]).cuda()
    mask = mask[f_idx][o_idx].astype(np.uint8)
    mask_tensor = torch.from_numpy(mask).cuda()

    # Extract unique objects from the mask, excluding zero
    unique_objects = np.unique(mask)
    unique_objects = unique_objects[unique_objects != 0].tolist()

    # Process the image and mask
    if init:
        processor.clear_memory()
        output_prob = processor.step(image_tensor, mask_tensor, objects=unique_objects)
    else:
        output_prob = processor.step(image_tensor)

    # Convert output probabilities to mask
    out_mask_tensor = processor.output_prob_to_mask(output_prob)

    # Convert the mask tensor to a NumPy array
    out_mask = out_mask_tensor.detach().cpu().numpy()

    # Refine the output mask using clip render mask
    out_mask[clip_render_mask[f_idx][o_idx] == 0] = 0

    return out_mask
def get_bounding_boxes_batch(render_masks, pad_rel=0.00):
    """
    Get bounding boxes for a batch of masks.

    Parameters:
    render_masks (np.ndarray): 3D input array of shape (batch_size, height, width).

    Returns:
    torch.Tensor: Batch of bounding boxes in xyxy format on CUDA.
    """
    batch_size = render_masks.shape[0]
    bounding_boxes = []

    for i in range(batch_size):
        d_model = render_masks[i]
        bounding_box = get_bounding_box(d_model, pad_rel=0.00)
        bounding_boxes.append(bounding_box)

    # Stack all bounding boxes into a single tensor
    bounding_boxes = torch.cat(bounding_boxes, dim=0).cuda()

    return bounding_boxes


def get_bounding_box(d_model, pad_rel=0.00):
    """
    Get the bounding box of the non-zero elements in the input array with optional padding.

    Parameters:
    d_model (np.ndarray): 2D input array.
    pad_rel (float): Relative padding (default: 5%)

    Returns:
    torch.Tensor: Padded bounding box in xyxy format on CUDA.
    """
    # Find the indices of non-zero elements
    non_zero_indices = np.nonzero(d_model)

    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        x_min = y_min = x_max = y_max = 0
    else:
        # Get the bounding box coordinates
        y_min, x_min = np.min(non_zero_indices, axis=1)
        y_max, x_max = np.max(non_zero_indices, axis=1)

    # Calculate padding
    x_pad = pad_rel * (x_max - x_min)
    y_pad = pad_rel * (y_max - y_min)

    # Get image dimensions
    height, width = d_model.shape

    # Create padded bounding box
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(width - 1, x_max + x_pad)
    y2 = min(height - 1, y_max + y_pad)

    # Create the bounding box in xyxy format
    bounding_box = torch.from_numpy(np.array([x1, y1, x2, y2])[None]).cuda()

    return bounding_box


def load_models(save_path):
    with open(save_path, 'rb') as f:
        object_models = pickle.load(f)
    print(f"Models loaded from {save_path}")
    return object_models


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clips_dir",
        default='/data/pose/dataset/bop/hot3d/train_aria',
        type=str,
        help="Path to a folder with clips. [train_aria, train_quest3]",
        )
    parser.add_argument(
        "--object_models_dir",
        default='/data/pose/dataset/bop/hot3d/object_models_eval',
        type=str,
        help="Path to a folder with 3D object models.",
        )
    parser.add_argument(
        "--mano_model_dir",
        type=str,
        help="Path to a folder with MANO model (MANO_RIGHT/LEFT.pkl files).",
        default="",
        )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="debug",
        )
    parser.add_argument(
        "--undistort",
        default=False,
        action="store_true",
        help="Whether to undistort images.",
        )
    parser.add_argument(
        "--hand_type",
        type=str,
        default="umetrack",
        help="Type of hand annotations to visualize ('umetrack' or 'mano').",
        )
    parser.add_argument(
        "--clip_start",
        type=int,
        help="ID of the first clip to visualize.",
        default=0,
        )
    parser.add_argument(
        "--clip_end",
        type=int,
        help="ID of the last clip to visualize.",
        default=-1,
        )
    parser.add_argument(
        "--frame_start",
        type=int,
        help="ID of the first clip to visualize.",
        default=0,
        )
    parser.add_argument(
        "--frame_end",
        type=int,
        help="ID of the last clip to visualize.",
        default=-1,
        )
    parser.add_argument(
        "--output_dir",
        default='../visualization/230731_save_img_sam',
        type=str,
        help="Path to a folder where to save visualizations.",
        )

    parser.add_argument(
        "--conf_thres",
        default=80,
        type=int,
        help='threshold of confidence',
        )
    parser.add_argument(
        "--iou_thres",
        default=60,
        type=int,
        help='threshold of confidence',
        )
    parser.add_argument(
        "--target_object",
        default=None,
        type=int,
        help='choose target object',
        )

    return parser.parse_args()


def calculate_visibility_iou(mask, d_model):
    # Ensure that the input shapes match
    assert mask.shape == d_model.shape, "The shapes of the two masks do not match"

    if len(mask.shape) == 2:  # Single mask case
        intersection = np.logical_and(mask, d_model)
        union = np.logical_or(mask, d_model)
        iou = np.round(np.sum(intersection) / (np.sum(union) + 1e-6) * 100.0).astype(np.uint8)
        return iou

    elif len(mask.shape) == 3:  # Multiple masks case
        num_masks = mask.shape[0]
        ious = []
        for i in range(num_masks):
            intersection = np.logical_and(mask[i], d_model[i])
            union = np.logical_or(mask[i], d_model[i])
            iou = np.round(np.sum(intersection) / (np.sum(union) + 1e-6) * 100.0).astype(np.uint8)
            ious.append(iou)

        return np.array(ious)
    else:
        raise ValueError("Unsupported mask shape")


def calculate_iou(prediction1, prediction2):
    intersection = np.logical_and(prediction1, prediction2)
    union = np.logical_or(prediction1, prediction2)
    iou = np.round(np.sum(intersection) / (np.sum(union) + 1e-6) * 100.0).astype(np.uint8)
    return iou


def calculate_l2_distance(track1, track2):
    return np.sqrt(np.sum((track1 - track2) ** 2, axis=-1))


def correct_depth_values(d_est, d_ren, mask, residual_threshold=None, max_trials=1000, stop_probability=0.99,
                         top_percent=0.5, use_polynomial_features=False, polynomial_degree=2):
    """
    Optimize estimated depth values using the RANSAC regression model with rendered depth values as ground truth.
    Optionally fits a polynomial of the specified degree.

    Parameters:
    - d_est: ndarray, estimated depth values of shape (h, w).
    - d_ren: ndarray, rendered depth values of shape (h, w) used as ground truth.
    - mask: ndarray, binary mask of shape (h, w), where 1 indicates valid regions and 0 indicates invalid regions.
    - residual_threshold: float, optional, maximum residual for a data sample to be classified as an inlier.
    - max_trials: int, optional, maximum number of iterations for random sample selection.
    - stop_probability: float, optional, stop iteration if at least this number of inliers are found with the desired probability.
    - top_percent: float, optional, top percentage of pairs with the smallest absolute differences to use.
    - use_polynomial_features: bool, optional, whether to use polynomial features.
    - polynomial_degree: int, optional, degree of the polynomial to fit.

    Returns:
    - d_est_corrected: ndarray, corrected depth values of shape (h, w).
    - coeff_ransac: list, coefficients of the fitted RANSAC model.
    - inlier_perc: float, percentage of inliers.
    - mad: float, median absolute deviation (if residual_threshold was None).
    """

    h, w = d_est.shape

    # Apply mask to the depth arrays
    valid_indices = mask.flatten() == 1
    d_est_flatten = d_est.flatten()[valid_indices].reshape(-1, 1)
    d_ren_flatten = d_ren.flatten()[valid_indices].reshape(-1, 1)

    # Calculate absolute differences and sort them
    abs_diffs = np.abs(d_est_flatten - d_ren_flatten)
    sorted_indices = np.argsort(abs_diffs, axis=0)

    # Select top_percent of pairs with the smallest absolute differences
    num_selected = int(len(d_est_flatten) * top_percent)
    selected_indices = sorted_indices[:num_selected].flatten()

    # Prepare the design matrix A with the selected estimated depths
    if use_polynomial_features:
        poly = PolynomialFeatures(degree=polynomial_degree)
        A_selected = poly.fit_transform(d_est_flatten[selected_indices])
    else:
        A_selected = np.hstack([d_est_flatten[selected_indices], np.ones((num_selected, 1))])

    b_selected = d_ren_flatten[selected_indices]

    # Initialize the RANSAC regressor with a linear model
    model_ransac = linear_model.RANSACRegressor(
        estimator=linear_model.LinearRegression(fit_intercept=False),
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        stop_probability=stop_probability
        )

    # Fit the RANSAC model to the selected data
    model_ransac.fit(A_selected, b_selected)

    # Calculate the number of inliers and their percentage
    inlier_count = model_ransac.inlier_mask_.sum()
    inlier_perc = 100 * inlier_count / float(model_ransac.inlier_mask_.size)
    coeff_ransac = model_ransac.estimator_.coef_.squeeze().tolist()
    print('Coefficients of the fitted model (RANSAC):', coeff_ransac)
    print('Inliers fraction: {:.2f}%'.format(inlier_perc))

    # Use the model to correct the estimated depth values
    if use_polynomial_features:
        A_all = poly.transform(d_est_flatten)
    else:
        A_all = np.hstack([d_est_flatten, np.ones((len(d_est_flatten), 1))])

    d_est_corrected_flatten = model_ransac.predict(A_all)

    # Calculate the MAD if the residual threshold was not provided
    if residual_threshold is None:
        residuals = np.abs(b_selected - model_ransac.predict(A_selected))
        mad = median_absolute_error(b_selected, model_ransac.predict(A_selected))
        # print('residuals:', residuals)
        print('Median Absolute Deviation (MAD):', mad)
    else:
        mad = None

    # Reshape the corrected depth values back to the original shape
    d_est_corrected = np.zeros(h * w)
    d_est_corrected[valid_indices] = d_est_corrected_flatten.flatten()
    d_est_corrected = d_est_corrected.reshape(h, w)

    return d_est_corrected, coeff_ransac, inlier_perc, mad


def calculate_depth_diff(est_depth, render_depth_all):
    """
    Calculate the visualized depth difference between estimated depth and rendered depth.

    Parameters:
    - est_depth: numpy array, estimated depth map
    - render_depth_all: numpy array, rendered depth map
    - cmap: colormap function to apply

    Returns:
    - vis_depth_diff: numpy array, visualized depth difference
    - valid_mask: numpy array, mask of valid depth values
    - diff_value_1000: numpy array, depth difference values divided by 1000
    """
    valid_mask = (est_depth > 0) & (render_depth_all > 0)

    depth_diff = valid_mask * abs(render_depth_all.astype(np.float32) - est_depth)

    depth_diff_valid = depth_diff[valid_mask]
    diff_value = depth_diff_valid
    depth_diff[valid_mask == False] = 0

    return valid_mask, diff_value, depth_diff


def calculate_diff_metrics(before, after, VALUE_SPLIT=1.0):
    min_before = np.round((before / VALUE_SPLIT).min(), 0)
    max_before = np.round((before / VALUE_SPLIT).max(), 0)
    mean_before = np.round((before / VALUE_SPLIT).mean(), 0)

    min_after = np.round((after / VALUE_SPLIT).min(), 0)
    max_after = np.round((after / VALUE_SPLIT).max(), 0)
    mean_after = np.round((after / VALUE_SPLIT).mean(), 0)

    return {
        'min_before': min_before,
        'max_before': max_before,
        'mean_before': mean_before,
        'min_after': min_after,
        'max_after': max_after,
        'mean_after': mean_after
        }


def process_and_visualize_depth(est_depth, render_depth, title_before, title_after, DEPTH_MIN_DIFF=0,
                                DEPTH_MAX_DIFF=200, RATIO_PERCENT=0.5, VALUE_SPLIT=1.0):
    vis_depth, valid_mask_depth, diff_depth_value, diff_depth = calculate_depth_diff(est_depth, render_depth,
                                                                                     DEPTH_MIN_DIFF, DEPTH_MAX_DIFF)
    try:
        corrected_depth, coeff_ransac, inlier_perc, mad = correct_depth_values(est_depth, render_depth,
                                                                               valid_mask_depth,
                                                                               top_percent=RATIO_PERCENT)
    except:
        corrected_depth = est_depth

    vis_corrected_depth, valid_mask_corrected_depth, diff_correct_depth_value, diff_correct_depth = calculate_depth_diff(
        corrected_depth, render_depth, DEPTH_MIN_DIFF, DEPTH_MAX_DIFF)

    value = calculate_diff_metrics(diff_depth_value, diff_correct_depth_value, VALUE_SPLIT)

    return (title_before, title_after, diff_depth, diff_correct_depth, value, value)


def visualize_depth_diff(results, vis_path, DEPTH_MIN_DIFF=0, DEPTH_MAX_DIFF=200):
    num_results = len(results)
    fig, axs = plt.subplots(num_results, 2, figsize=(20, 7.5 * num_results))

    if num_results == 1:
        axs = [axs]  # Ensure axs is always a list of axes

    for i, (title_before, title_after, depth_diff, correct_depth_diff, value_before, value_after) in enumerate(results):
        # Rotate the images
        depth_diff_rot = np.rot90(depth_diff, k=3)
        correct_depth_diff_rot = np.rot90(correct_depth_diff, k=3)

        # Plot the depth difference before correction
        im0 = axs[i][0].imshow(depth_diff_rot, vmin=DEPTH_MIN_DIFF, vmax=DEPTH_MAX_DIFF, cmap='jet')
        axs[i][0].xaxis.set_label_position('top')
        axs[i][0].xaxis.tick_top()
        axs[i][0].set_title(title_before, fontsize=20)
        axs[i][0].axis('off')
        cbar0 = plt.colorbar(im0, ax=axs[i][0], shrink=0.8)
        cbar0.ax.tick_params(labelsize=10)
        cbar0.set_label('Depth (mm)', fontsize=12)
        axs[i][0].text(0.5, -0.05,
                       f'Min({value_before["min_before"]:.0f} mm)    Max({value_before["max_before"]:.0f} mm)     Mean({value_before["mean_before"]:.0f} mm)',
                       ha='center', va='top', transform=axs[i][0].transAxes, fontsize=20)

        # Plot the depth difference after correction
        im1 = axs[i][1].imshow(correct_depth_diff_rot, vmin=DEPTH_MIN_DIFF, vmax=DEPTH_MAX_DIFF, cmap='jet')
        axs[i][1].xaxis.set_label_position('top')
        axs[i][1].xaxis.tick_top()
        axs[i][1].set_title(title_after, fontsize=20)
        axs[i][1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axs[i][1], shrink=0.8)
        cbar1.ax.tick_params(labelsize=10)
        cbar1.set_label('Depth (mm)', fontsize=12)
        axs[i][1].text(0.5, -0.05,
                       f'Min({value_after["min_after"]:.0f} mm)   Max({value_after["max_after"]:.0f} mm)    Mean({value_after["mean_after"]:.0f} mm)',
                       ha='center', va='top', transform=axs[i][1].transAxes, fontsize=20)

    plt.tight_layout()
    plt.savefig(vis_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.close(fig)


def update_depth_and_mask_and_rgb(render_depth_all: np.ndarray, ren_obj_depth: np.ndarray,
                                  render_mask_all: np.ndarray, ren_obj_mask: np.ndarray,
                                  render_rgb_all: np.ndarray, ren_obj_rgb: np.ndarray):
    """Update render depth, mask, and RGB arrays with object depth, mask, and RGB arrays.

    Args:
        render_depth_all (np.ndarray): The render depth array to be updated.
        ren_obj_depth (np.ndarray): The object depth array.
        render_mask_all (np.ndarray): The render mask array to be updated.
        ren_obj_mask (np.ndarray): The object mask array.
        render_rgb_all (np.ndarray): The render RGB array to be updated.
        ren_obj_rgb (np.ndarray): The object RGB array.

    Returns:
        tuple: Updated render depth array, render mask array, and render RGB array.
    """
    mask_zero = (render_depth_all == 0)
    non_mask_zero = np.logical_and(
        np.logical_and((render_depth_all != 0), (ren_obj_depth != 0)),
        render_depth_all > ren_obj_depth
        )

    render_depth_all[non_mask_zero] = ren_obj_depth[non_mask_zero]
    render_depth_all[mask_zero] = ren_obj_depth[mask_zero]

    render_mask_all[non_mask_zero] = ren_obj_mask[non_mask_zero]
    render_mask_all[mask_zero] = ren_obj_mask[mask_zero]

    render_rgb_all[non_mask_zero] = ren_obj_rgb[non_mask_zero]
    render_rgb_all[mask_zero] = ren_obj_rgb[mask_zero]

    return render_depth_all, render_mask_all, render_rgb_all


def update_depth_and_mask(render_depth_all: np.ndarray, ren_obj_depth: np.ndarray,
                          render_mask_all: np.ndarray,
                          ren_obj_mask: np.ndarray):
    """Update render depth and mask arrays with object depth and mask arrays.

    Args:
        render_depth_all (np.ndarray): The render depth array to be updated.
        ren_obj_depth (np.ndarray): The object depth array.
        render_mask_all (np.ndarray): The render mask array to be updated.
        ren_obj_mask (np.ndarray): The object mask array.

    Returns:
        np.ndarray: Updated render depth array.
        np.ndarray: Updated render mask array.
    """
    mask_zero = (render_depth_all == 0)
    non_mask_zero = np.logical_and(
        np.logical_and((render_depth_all != 0), (ren_obj_depth != 0)),
        render_depth_all > ren_obj_depth
        )

    render_depth_all[non_mask_zero] = ren_obj_depth[non_mask_zero]
    render_depth_all[mask_zero] = ren_obj_depth[mask_zero]
    render_mask_all[non_mask_zero] = ren_obj_mask[non_mask_zero]
    render_mask_all[mask_zero] = ren_obj_mask[mask_zero]

    return render_depth_all, render_mask_all


def get_number_of_frames(tar: Any) -> int:
    """Returns the number of frames in a clip.

    Args:
        tar: File handler of an open tar file with clip data.
    Returns:
        Number of frames in the given tar file.
    """

    max_frame_id = -1
    for x in tar.getnames():
        if x.endswith(".info.json"):
            frame_id = int(x.split(".info.json")[0])
            if frame_id > max_frame_id:
                max_frame_id = frame_id
    return max_frame_id + 1


def load_image(
        tar: Any,
        frame_key: str,
        stream_key: str,
        dtype: np.typename = np.uint8
        ) -> np.ndarray:
    """Loads an image from the specified frame and stream of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame from which to load the image.
        stream_key: Key of the stream from which to load the image.
        dtype: Desired type of the loaded image.
    Returns:
        Numpy array with the loaded image.
    """

    file = tar.extractfile(f"{frame_key}.image_{stream_key}.jpg")
    return imageio.imread(file).astype(dtype)


def load_cameras(
        tar: Any,
        frame_key: str,
        ) -> Dict[str, camera.CameraModel]:
    """Loads cameras for all image streams in a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the cameras.
    Returns:
        A dictionary mapping a stream key to a camera model.
    """

    cameras_raw = json.load(tar.extractfile(f"{frame_key}.cameras.json"))

    cameras = {}
    for stream_key, camera_raw in cameras_raw.items():
        cameras[stream_key] = camera.from_json(camera_raw)

    return cameras


def load_object_annotations(
        tar: Any,
        frame_key: str,
        ) -> Optional[Dict[str, Any]]:
    """Loads object annotations for a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the annotations.
    Returns:
        A dictionary with object annotations.
    """

    filename = f"{frame_key}.objects.json"
    if filename in tar.getnames():
        return json.load(tar.extractfile(filename))
    else:
        # Annotations are not provided for test clips.
        None


def load_hand_annotations(
        tar: Any,
        frame_key: str,
        ) -> Optional[Dict[str, Any]]:
    """Loads hand annotations for a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the annotations.
    Returns:
        A dictionary with hand annotations. Poses are provided in two
        formats: UmeTrack and MANO.
    """

    filename = f"{frame_key}.hands.json"
    if filename in tar.getnames():
        return json.load(tar.extractfile(filename))
    else:
        # Annotations are not provided for test clips.
        None


def load_hand_shape(
        tar: Any
        ) -> Optional[HandShapeCollection]:
    """Loads hand shape for a specified clip.

    Args:
        tar: File handler of an open tar file with clip data.
    Returns:
        Hand shape in two formats: UmeTrack and MANO.
    """

    filename = "__hand_shapes.json__"
    if filename in tar.getnames():
        shape_params_dict = json.load(tar.extractfile(filename))
        return HandShapeCollection(
            mano_beta=torch.tensor(shape_params_dict["mano"]),
            umetrack=from_umetrack_hand_model_json(
                shape_params_dict["umetrack"]
                )
            )
    else:
        # Hand shapes are not provided for some test clips.
        return None


def get_hand_meshes(
        hands: Dict[str, Any],
        hand_shape: HandShapeCollection,
        hand_type: str = "umetrack",
        mano_model: Optional[MANOHandModel] = None,
        ) -> Dict[str, trimesh.Trimesh]:
    """Provides hand meshes of specified shape and poses.

    Args:
        hands: Hand annotations (including hand poses).
        hand_shape: Hand shape.
        hand_type: Hand type ("umetrack" or "mano").
        mano_model: MANO hand model (needs to be provided
            if hand_type == "mano").
    Returns:
        Triangular meshes of left and/or right hands.
    """

    if hand_type == "mano" and mano_model is None:
        raise ValueError("MANO hand model is missing.")

    hand_poses = decode_hand_pose(hands)

    meshes: Dict[HandSide, trimesh.Trimes] = {}
    for hand_side, hand_pose in hand_poses.items():
        _, hand_verts, hand_faces = visualization.get_keypoints_and_mesh(
            hand_pose=hand_pose,
            hand_shape=hand_shape,
            mano_model=mano_model,
            pose_type=hand_type,
            )

        meshes[hand_side] = trimesh.Trimesh(
            vertices=hand_verts,
            faces=hand_faces,
            process=False,
            )

    return meshes


def load_mesh(
        path: str,
        ) -> trimesh.Trimesh:
    """Loads a 3D mesh model from a specified path.

    Args:
        path: Path to the model to load.
    Returns.
        Loaded mesh.
    """

    # Load the scene.
    scene = trimesh.load_mesh(
        path,
        process=True,
        merge_primitives=True,
        skip_materials=True,
        )

    # Represent the scene by a single mesh.
    mesh = scene.dump(concatenate=True)

    # Make sure there are no large triangles (the rasterizer
    # from hand_tracking_toolkit becomes slow if some triangles
    # are much larger than others)
    mesh = subdivide_mesh(mesh)

    # Clean the mesh.
    mesh.process(validate=True)

    return mesh


def subdivide_mesh(
        mesh: trimesh.Trimesh,
        max_edge: float = 0.01,
        max_iters: int = 100,
        debug: bool = False
        ):
    """Subdivides mesh such as all edges are shorter than a threshold.

    Args:
        mesh: Mesh to subdivide.
        max_edge: Maximum allowed edge length in meters (note that this may
            not be reachable if max_iters is too low).
        max_iters: Number of subdivision iterations.
    Returns.
        Subdivided mesh.
    """

    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge, max_iter=max_iters,
        )
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    if debug:
        print(f"Remeshing: {len(mesh.vertices)} -> {len(new_mesh.vertices)}")

    return new_mesh


def convert_to_pinhole_camera(
        camera_model: camera.CameraModel,
        focal_scale: float = 1.0
        ) -> camera.CameraModel:
    """Converts a camera model to a pinhole version.

    Args:
        camera_model: Input camera model.
        focal_scale: Focal scaling factor (can be used to contol
            the portion of an original fisheye image that is seen in
            the resulting pinhole camera).
    Returns:
        Pinhole camera model.
    """

    return camera.PinholePlaneCameraModel(
        width=camera_model.width,
        height=camera_model.height,
        f=[camera_model.f[0] * focal_scale, camera_model.f[1] * focal_scale],
        c=camera_model.c,
        distort_coeffs=[],
        T_world_from_eye=camera_model.T_world_from_eye,
        )


def se3_from_dict(
        se3_dict: Dict[str, Any]
        ) -> np.ndarray:
    """Converts a dictionary to an 4x4 SE3 transformation matrix.

    Args:
        se3_dict: Dictionary with items "quaternion_wxyz" and
            "translation_xyz".
    Returns:
        4x4 numpy array with a 4x4 SE3 transformation matrix.
    """

    return math_utils.quat_trans_to_matrix(
        *se3_dict["quaternion_wxyz"],
        *se3_dict["translation_xyz"],
        )


def stack_images(
        images: Sequence[np.ndarray],
        axis: int = 1
        ) -> np.ndarray:
    """Stack a list of images along a specified axis.

    Args:
        images: List of images to stack.
        axis: Axis along which to stack the images (0 for vertical, 1 for horizontal).
    Returns:
        Input images stacked into a single image
        (if stacking horizontally and the height of the images is different, all are
        resized to the smallest height; if stacking vertically and the width of the
        images is different, all are resized to the smallest width).
    """

    if axis == 1:  # Horizontal stacking
        min_image_height = images[0].shape[0]
        all_same_height = True
        for image in images[1:]:
            if image.shape[0] < min_image_height:
                min_image_height = image.shape[0]
                all_same_height = False

        if not all_same_height:
            for image_id, image in enumerate(images):
                scale = min_image_height / image.shape[0]
                images[image_id] = cv2.resize(
                    image, (int(scale * image.shape[1]), min_image_height)
                    )

    elif axis == 0:  # Vertical stacking
        min_image_width = images[0].shape[1]
        all_same_width = True
        for image in images[1:]:
            if image.shape[1] < min_image_width:
                min_image_width = image.shape[1]
                all_same_width = False

        if not all_same_width:
            for image_id, image in enumerate(images):
                scale = min_image_width / image.shape[1]
                images[image_id] = cv2.resize(
                    image, (min_image_width, int(scale * image.shape[0]))
                    )

    stacked_image = np.concatenate(images, axis=axis)
    return stacked_image.astype(np.uint8)


def vis_mask_contours(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[float] = (255, 255, 255),
        thickness: int = 1
        ) -> np.ndarray:
    """Overlays mask contour on top of an image.

    Args:
        image: Base image.
        mask: Mask whose contour will be overlaid on the image.
        color: Color of the contour.
        thickness: Thickness of the contour.
    Returns:
        Image overlaid with the mask contour.
    """

    contours = cv2.findContours(
        mask.astype(np.uint8),
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
        )[0]

    return cv2.drawContours(
        image, contours, -1, color, thickness, cv2.LINE_AA
        )
