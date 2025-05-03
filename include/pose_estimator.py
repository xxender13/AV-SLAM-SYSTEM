# src/pose_estimator.py

import os
import cv2
import numpy as np

# --- Load intrinsics from calib.txt ---
def load_intrinsics(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    P_rect = []
    for i, line in enumerate(lines):
        if line.startswith('P_rect_02:'):
            first_line = list(map(float, line.strip().split()[1:]))
            second_line = list(map(float, lines[i + 1].strip().split()))
            third_line = list(map(float, lines[i + 2].strip().split()))
            P_rect = first_line + second_line + third_line
            break
    if len(P_rect) != 12:
        raise ValueError("Could not parse 3x4 projection matrix from calib.txt")
    fx = P_rect[0]
    fy = P_rect[5]
    cx = P_rect[2]
    cy = P_rect[6]
    return fx, fy, (cx, cy)

# --- Get frame index pairs based on strategy ---
def get_frame_pairs(strategy, total_frames):
    if strategy == '1-2-3-4':
        return [(i, i+1) for i in range(total_frames - 1)]
    elif strategy == '1-3-5-7':
        return [(i, i+2) for i in range(0, total_frames - 2, 2)]
    elif strategy == '1-5-9-13':
        return [(i, i+4) for i in range(0, total_frames - 4, 4)]
    else:
        raise ValueError("Invalid match strategy.")

# --- Extract top-100 matched keypoints using ORB ---
def keypoint_extraction(img1, img2):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return [], []
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = [kp1[m.queryIdx].pt for m in matches[:100]]
    pts2 = [kp2[m.trainIdx].pt for m in matches[:100]]
    return np.int32(pts1), np.int32(pts2)

# --- Main pose estimation pipeline ---
def estimate_and_save_poses(image_dir, calib_path, output_root, match_strategy='1-3-5-7'):
    fx, fy, (cx, cy) = load_intrinsics(calib_path)
    focal_length = (fx + fy) / 2
    principal_point = (cx, cy)

    out_dir = os.path.join(output_root, "pose")
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, 'poses.txt')
    open(out_txt, 'w').close()  # Clear previous results

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    total_frames = len(image_files)
    frame_pairs = get_frame_pairs(match_strategy, total_frames)

    with open(out_txt, 'a') as f:
        f.write("Pose Estimations (Rotation R and Translation t between frames)\n")
        f.write("=" * 60 + "\n")

        for i, j in frame_pairs:
            img1_path = os.path.join(image_dir, image_files[i])
            img2_path = os.path.join(image_dir, image_files[j])
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                continue

            pts1, pts2 = keypoint_extraction(img1, img2)
            if len(pts1) < 8 or len(pts2) < 8:
                continue

            # Essential matrix and pose recovery
            E, _ = cv2.findEssentialMat(pts1, pts2, focal=focal_length, pp=principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=focal_length, pp=principal_point)

            # Write results
            f.write(f"Frame {i} to {j}\n")
            f.write("Rotation Matrix R:\n")
            f.write(np.array2string(R, precision=4, separator=', ') + "\n")
            f.write("Translation Vector t:\n")
            f.write(np.array2string(t.T, precision=4, separator=', ') + "\n")
            f.write("-" * 60 + "\n")

    print(f"✅ Pose Estimation Complete. Results saved to: {out_txt}")
