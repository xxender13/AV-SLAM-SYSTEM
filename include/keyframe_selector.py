# include/keyframe_selector.py

import cv2
import os

def select_visual_keyframes(image_dir, match_threshold=40, step=1, verbose=True):
    """
    Select keyframes based on ORB descriptor matching between grayscale images.

    Args:
        image_dir (str): Path to grayscale images.
        match_threshold (int): If number of good matches < threshold, mark as keyframe.
        step (int): Frame step size (default 1).
        verbose (bool): If True, prints matching info.

    Returns:
        keyframes (List[int]): List of selected keyframe indices.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    keyframes = [0]

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ref_idx = 0
    ref_img = cv2.imread(os.path.join(image_dir, image_files[ref_idx]), cv2.IMREAD_GRAYSCALE)
    _, ref_desc = orb.detectAndCompute(ref_img, None)

    for i in range(1, len(image_files), step):
        curr_img = cv2.imread(os.path.join(image_dir, image_files[i]), cv2.IMREAD_GRAYSCALE)
        _, curr_desc = orb.detectAndCompute(curr_img, None)

        if ref_desc is None or curr_desc is None:
            continue

        matches = bf.match(ref_desc, curr_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]

        if verbose:
            print(f"Frame {i}: {len(good_matches)} good matches")

        if len(good_matches) < match_threshold:
            keyframes.append(i)
            ref_idx = i
            ref_img = curr_img
            ref_desc = curr_desc

    if verbose:
        print(f"\nSelected {len(keyframes)} keyframes out of {len(image_files)} total.")

    return keyframes
