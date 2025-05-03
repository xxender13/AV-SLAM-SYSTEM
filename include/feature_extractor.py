# src/feature_extractor.py
import os
import cv2


def extract_and_match_features(image_dir, output_root, fast_thresh=25, orb_features=300):
    out_dir = os.path.join(output_root, "features")
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    fast = cv2.FastFeatureDetector_create(threshold=fast_thresh)
    orb = cv2.ORB_create(nfeatures=orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_dict, des_dict, img_dict = {}, {}, {}

    for idx, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        fast_kp = fast.detect(img)
        orb_kp, orb_des = orb.detectAndCompute(img, None)

        for kp in fast_kp:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(color, (x, y), 2, (255, 0, 0), -1)
        if orb_kp:
            for kp in orb_kp:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(color, (x, y), 2, (0, 255, 0), -1)

        cv2.imwrite(os.path.join(out_dir, f"features_{idx:04d}.png"), color)
        kp_dict[idx], des_dict[idx], img_dict[idx] = orb_kp, orb_des, img

    for i in range(0, len(image_files) - 1, 2):
        des1, des2 = des_dict.get(i), des_dict.get(i + 2)
        kp1, kp2 = kp_dict.get(i), kp_dict.get(i + 2)
        img1, img2 = img_dict.get(i), img_dict.get(i + 2)
        if any(x is None for x in (des1, des2, kp1, kp2)):
            continue

        matches = matcher.match(des1, des2)
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
        cv2.imwrite(os.path.join(out_dir, f"match_{i:04d}_{i + 2:04d}.png"), match_img)

    print(f"✅ Feature extraction and matching complete. Saved to: {out_dir}")
