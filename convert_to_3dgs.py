import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time
import shutil 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fast3r.dust3r.utils.image import load_images_ex
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.viz.video_utils import extract_frames_from_video
from fast3r.utils.sfm_utils import (get_sorted_image_files, save_extrinsic, save_intrinsics, save_points3D, export_combined_ply)

def main(input_dir, output_dir, image_size, checkpoint_dir, min_conf_thr, use_align):
    
    # 1. Check if it's necessary to extract iamges from the video and load images
    image_dir = str(Path(output_dir) / "images")
    sparse_dir = str(Path(output_dir) / "sparse/0")
    if not os.path.exists(sparse_dir):
            os.makedirs(sparse_dir)
    image_files = ""
    image_suffix = ".jpg"
    if input_dir.lower().endswith(".mp4"):
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir)    
        image_files = extract_frames_from_video(video_path=input_dir, output_dir=image_dir)
    else:
        image_dir = Path(input_dir) / 'images'
        image_files, image_suffix = get_sorted_image_files(image_dir = image_dir)

    images, org_imgs_shape, new_imgs_shape = load_images_ex(image_files, size=image_size)

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fast3R.from_pretrained(checkpoint_dir).to(device)
    # Create a lightweight lightning module wrapper for the model.
    # This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
    lit_model = MultiViewDUSt3RLitModule.load_for_inference(model)
    # Set model to evaluation mode
    model.eval()
    lit_model.eval()

    # 3. Run inference and estimate camera poses
    print(f">> Fast3R inference ...")
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32, # or use torch.bfloat16 if supported
        verbose=True,
        profiling=True,
    )
    
    print(f">> Estimate poses and focal ...")
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    poses_w2c = [np.linalg.inv(item) for item in poses_c2w_batch[0]]
    focals = estimated_focals[0]

    # Align points
    if use_align:
        lit_model.align_local_pts3d_to_global(
            preds=output_dict['preds'],
            views=output_dict['views'],
            min_conf_thr_percentile=min_conf_thr
        )

    # 4. Save results
    print(f'>> Saving results ...')
    save_extrinsic(sparse_dir, poses_w2c, image_files, image_suffix)
    save_intrinsics(sparse_dir, focals, org_imgs_shape, new_imgs_shape, save_focals=True)
    if use_align:
        export_combined_ply(output_dict['preds'], output_dict['views'], sparse_dir, 'pts3d_local_aligned_to_global', conf_key_to_visualize='conf_local', min_conf_thr_percentile=min_conf_thr)
    else:
        export_combined_ply(output_dict['preds'], output_dict['views'], sparse_dir, 'pts3d_in_other_view', conf_key_to_visualize='conf', min_conf_thr_percentile=min_conf_thr)
    print(f'>> Fast3R Reconstruction is successfully converted to COLMAP files in: {output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Fast3r Reconstruction to COLMAP files ...")
    parser.add_argument('--checkpoint_dir', type=str, default="jedyang97/Fast3R_ViT_Large_512")
    parser.add_argument('--image_size', type=int, default=224, help="Size to resize images, 224 or 512")
    # parser.add_argument('--input_dir', '-s', type=str, default="/data/lurvelly/Louis/Datasets/tandt_db/tandt/truck", help='Directory containing images or video')
    parser.add_argument('--input_dir', '-s', type=str, default="demo_examples/family/Family.mp4", help='Directory containing images or video')
    parser.add_argument('--output_dir', '-o', type=str, default="fast3r_colmap", help='Directory to store the results')
    parser.add_argument('--min_conf_thr', type=int, default=50, help="percentile between 0 - 100")
    parser.add_argument('--use_align', type=bool, default=False, help="Whether use align_local_pts3d_to_global method")
    args = parser.parse_args()

    # Using proxy for Chinese mainland users
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    main(input_dir=args.input_dir, output_dir=args.output_dir, image_size=args.image_size, checkpoint_dir=args.checkpoint_dir, min_conf_thr=args.min_conf_thr, use_align=args.use_align)