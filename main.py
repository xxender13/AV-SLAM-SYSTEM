# main.py
import os
from src.slam_pipeline import run_slam_pipeline

if __name__ == "__main__":
    dataset_dir = "./data/2011_09_26_drive_0051_sync/image_02/data"
    calib_file = "./data/2011_09_26_drive_0051_sync/calib.txt"
    output_dir = "output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run full SLAM pipeline
    run_slam_pipeline(dataset_dir, calib_file, output_dir)
