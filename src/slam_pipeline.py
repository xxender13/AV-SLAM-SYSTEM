# src/slam_pipeline.py
import os
from include.grayscale_converter import convert_to_grayscale
from include.feature_extractor import extract_and_match_features
from include.pose_estimator import estimate_and_save_poses
from include.trajectory_builder import build_and_plot_trajectory

from include.matcher import match_and_draw

import cv2

def run_slam_pipeline(image_dir, calib_path, output_root, match_step=2):
    # Step 1: Grayscale Conversion
    print("\n🔹 Step 1: Grayscale Conversion")
    convert_to_grayscale(image_dir, output_root)
    gray_dir = os.path.join(output_root, "grayscale")

    # Step 2: Feature Extraction & Basic Matching
    print("\n🔹 Step 2: Feature Extraction and Matching")
    extract_and_match_features(image_dir, output_root)

    # Step 3: Custom Matcher Visualization (new)
    print("\n🔹 Step 3: Visual Match Output Generation")
    matcher_dir = os.path.join(output_root, "matcher")
    os.makedirs(matcher_dir, exist_ok=True)

    gray_files = sorted([f for f in os.listdir(gray_dir) if f.endswith('.png')])
    for i in range(0, len(gray_files) - match_step, match_step):
        img1_path = os.path.join(gray_dir, gray_files[i])
        img2_path = os.path.join(gray_dir, gray_files[i + match_step])
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        tag = f"{i:04d}_{i+match_step:04d}"
        out_path = os.path.join(matcher_dir, f"match_{tag}.png")
        match_and_draw(img1, img2, out_path, tag)

    # Step 4: Pose Estimation
    print("\n🔹 Step 4: Pose Estimation")
    estimate_and_save_poses(image_dir, calib_path, output_root)

    # Step 5: Trajectory Build & Smoothing
    print("\n🔹 Step 5: Trajectory Building & Optimization")
    build_and_plot_trajectory(output_root)

    print("\n✅ Full SLAM pipeline completed and outputs saved in:", output_root)