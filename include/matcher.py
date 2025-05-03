# include/matcher.py
import cv2
import os

def match_and_draw(img1, img2, out_path, tag="match"):
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print(f"Skipping {tag}: not enough keypoints.")
        return None

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, match_img)
    print(f"✅ Saved matcher output to {out_path}")

    return match_img
