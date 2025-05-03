import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from include.keyframe_selector import select_visual_keyframes

def extract_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("Frame"):
            try:
                R = np.array(ast.literal_eval(''.join(lines[i+2:i+5])))
                t = np.array(ast.literal_eval(lines[i+6])).reshape(3, 1)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()
                poses.append(T)
            except:
                pass
            i += 8
        else:
            i += 1
    return poses

def build_trajectory(pose_list):
    T = np.eye(4)
    traj = []
    for pose in pose_list:
        T = T @ pose
        traj.append(T[:3, 3])
    return np.array(traj)

def smooth_trajectory(traj, window=7, poly=2):
    if len(traj) < window:
        return traj
    smoothed = np.zeros_like(traj)
    for dim in range(3):  # X, Y, Z
        smoothed[:, dim] = savgol_filter(traj[:, dim], window_length=window, polyorder=poly)
    return smoothed

def set_equal_axis(ax, traj1, traj2):
    combined = np.vstack((traj1, traj2))
    min_vals = combined.min(axis=0)
    max_vals = combined.max(axis=0)

    max_range = (max_vals - min_vals).max()
    mid_vals = (max_vals + min_vals) / 2

    ax.set_xlim(mid_vals[0] - max_range/2, mid_vals[0] + max_range/2)
    ax.set_ylim(mid_vals[1] - max_range/2, mid_vals[1] + max_range/2)
    ax.set_zlim(mid_vals[2] - max_range/2, mid_vals[2] + max_range/2)
    ax.set_box_aspect([1, 1, 1])

def build_and_plot_trajectory(output_root):
    pose_file = os.path.join(output_root, "pose", "poses.txt")
    traj_dir = os.path.join(output_root, "trajectory")
    os.makedirs(traj_dir, exist_ok=True)

    # Step 1: Build full trajectory
    poses = extract_poses(pose_file)
    base_traj = build_trajectory(poses)
    np.savetxt(os.path.join(traj_dir, "base_trajectory.txt"), base_traj)

    # Step 2: Select keyframes visually using ORB matching
    gray_dir = os.path.join(output_root, "grayscale")
    key_indices_all = select_visual_keyframes(gray_dir, match_threshold=40)
    valid_max_index = len(base_traj) - 1
    key_indices = [i for i in key_indices_all if i <= valid_max_index]
    optimized_traj = base_traj[key_indices]

    # Step 3: Smooth the trajectory
    optimized_traj = smooth_trajectory(optimized_traj, window=7, poly=2)
    np.savetxt(os.path.join(traj_dir, "smoothed_trajectory.txt"), optimized_traj)

    # Step 4: Plot base trajectory
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(*base_traj.T, color='blue', label='Base Trajectory')
    ax1.set_title("Base Trajectory")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_equal_axis(ax1, base_traj, optimized_traj)
    plt.savefig(os.path.join(traj_dir, "base_trajectory.png"))
    plt.close()

    # Step 5: Plot optimized smoothed trajectory
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(*optimized_traj.T, color='green', label='Smoothed Trajectory')
    ax2.set_title("Optimized Smoothed Trajectory (Visual Keyframes)")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    set_equal_axis(ax2, base_traj, optimized_traj)
    plt.savefig(os.path.join(traj_dir, "optimized_trajectory.png"))
    plt.close()

    print("✅ Base and optimized trajectories saved and plotted in:", traj_dir)
