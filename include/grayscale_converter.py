# src/grayscale_converter.py
import os
import cv2
import time


def convert_to_grayscale(image_dir, output_root):
    out_dir = os.path.join(output_root, "grayscale")
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    start = time.time()
    for f in image_files:
        path = os.path.join(image_dir, f)
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(out_dir, f), gray)
    end = time.time()

    print(f"✅ Grayscale conversion complete. Saved to: {out_dir}")
    print(f"⏱️ Runtime: {end - start:.2f} seconds")
