#!/usr/bin/env python3

import argparse
import os
import tarfile
import imageio
import numpy as np
import trimesh
from typing import Any, Dict, List, Optional

# HT toolkit (https://github.com/facebookresearch/hand_tracking_toolkit)
from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import HandShapeCollection, warp_image
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

import clip_util

import matplotlib
# matplotlib.use('tkagg')
import pickle

def vis_clip(
        clip_path: str,
        object_models: Dict[int, trimesh.Trimesh],
        hand_type: str,
        mano_model: Optional[MANOHandModel],
        undistort: bool,
        output_dir: str,
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

    # Load hand shape.
    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)

    # Per-frame visualization.
    for frame_id in range(clip_util.get_number_of_frames(tar)):
        print(f"Visualizing frame {frame_id}...")
        frame_key = f"{frame_id:06d}"

        # Load camera parameters.
        cameras = clip_util.load_cameras(tar, frame_key)

        # Available image streams.
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        # Load hand and object annotations.
        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(tar, frame_key)
        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(tar, frame_key)

        # Get hand meshes.
        hand_meshes: Dict[str, trimesh.Trimesh] = {}
        if hand_shape is not None and hands is not None:
            hand_meshes = clip_util.get_hand_meshes(hands, hand_shape, hand_type, mano_model)

        # Visualize hands and objects in each image of the current frame
        # (each Quest3 frame has 2 monochrome images while each Aria frame
        # has 1 RGB and 2 monochrome images).
        vis_images: List[np.ndarray] = []
        vis_depths = []
        vis_masks = []
        for stream_id in image_streams:
            stream_key = str(stream_id)

            # Load the image.
            image: np.ndarray = clip_util.load_image(tar, frame_key, stream_key)

            # Make sure the image has 3 channels.
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            # Camera parameters of the current image.
            camera_model = cameras[stream_id]

            # Optional undistortion (by warping the image to a pinhole camera).
            if undistort:
                camera_model_orig = camera_model
                camera_model = clip_util.convert_to_pinhole_camera(camera_model)
                image = warp_image(
                    src_camera=camera_model_orig,
                    dst_camera=camera_model,
                    src_image=image,
                    )

            # Visualize object contours.
            for instance_list in objects.values():
                for instance in instance_list:
                    bop_id = int(instance["object_bop_id"])

                    # Transformation from the model to the world space.
                    T = clip_util.se3_from_dict(
                        instance["T_world_from_object"]
                        )

                    # Vertices in the model space.
                    verts_in_m = object_models[bop_id].vertices

                    # Vertices in the world space (can be brought to the camera
                    # space by the inverse of camera_model.T_world_from_eye).
                    verts_in_w = (T[:3, :3] @ verts_in_m.T + T[:3, 3:]).T

                    # Render the object model (outputs: rgb, mask, depth).
                    ren_obj_rgb, ren_obj_mask, ren_obj_depth = rasterizer.rasterize_mesh(
                        verts=verts_in_w,
                        faces=object_models[bop_id].faces,
                        vert_normals=object_models[bop_id].vertex_normals,
                        camera=camera_model,
                        )

                    # Visualize the object contour on top of the image.
                    image = clip_util.vis_mask_contours(image, ren_obj_mask, (0, 255, 0))

            # Visualize hand contours.
            for hand_mesh in hand_meshes.values():
                # Render the hand model (outputs: rgb, mask, depth).
                ren_hand_rgb, ren_hand_mask, ren_hand_depth = rasterizer.rasterize_mesh(
                    verts=hand_mesh.vertices,
                    faces=hand_mesh.faces,
                    vert_normals=hand_mesh.vertex_normals,
                    camera=camera_model,
                    )

                # Visualize the hand contour on top of the image.
                image = clip_util.vis_mask_contours(image, ren_hand_mask, (255, 255, 255))

            # Orient the image upright.
            image = np.rot90(image, k=3)

            # Store the image.
            vis_images.append(image)

        # Stack all images from the current frame horizontally and save.
        vis_frame_path = os.path.join(clip_output_path, f"{frame_key}.jpg")
        imageio.imwrite(vis_frame_path, clip_util.stack_images(vis_images))



def load_models(save_path):
    with open(save_path, 'rb') as f:
        object_models = pickle.load(f)
    print(f"Models loaded from {save_path}")
    return object_models

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips_dir",
        type=str,
        help="Path to a folder with clips.",
        )
    parser.add_argument(
        "--object_models_dir",
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
        "--undistort",
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
        "--output_dir",
        type=str,
        help="Path to a folder where to save visualizations.",
        required=True,
        )
    args = parser.parse_args()

    # Make sure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load object models.
    save_path = '../object_models.pkl'

    if os.path.exists(save_path):
        object_models = load_models(save_path)
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
    for clip in clips:
        clip_id = int(clip.split(".tar")[0].split("clip-")[1])

        # Skip the clip if it is not in the specified range.
        if clip_id < args.clip_start or (
                args.clip_end >= 0 and clip_id > args.clip_end
        ):
            continue

        vis_clip(
            clip_path=os.path.join(args.clips_dir, clip),
            object_models=object_models,
            hand_type=args.hand_type,
            mano_model=mano_model,
            undistort=args.undistort,
            output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()