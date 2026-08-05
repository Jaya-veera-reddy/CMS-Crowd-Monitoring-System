import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import scipy.io
import numpy as np
from utils.generate_density_map import generate_density_map


def process_dataset(image_dir, gt_dir, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mat")])

    for img_name, gt_name in zip(img_files, gt_files):

        img_path = os.path.join(image_dir, img_name)
        gt_path = os.path.join(gt_dir, gt_name)

        img = cv2.imread(img_path)

        if img is None:
            print("Invalid image:", img_name)
            continue

        mat = scipy.io.loadmat(gt_path)

        points = mat["image_info"][0][0][0][0][0]

        density = generate_density_map(img.shape[:2], points)

        save_path = os.path.join(save_dir, img_name.replace(".jpg", ".npy"))

        np.save(save_path, density)

        print("Processed:", img_name)


# Part A
process_dataset(
    "dataset/ShanghaiTech/part_A/train_data/images",
    "dataset/ShanghaiTech/part_A/train_data/ground-truth",
    "dataset/ShanghaiTech/density_maps/part_A"
)

# Part B
process_dataset(
    "dataset/ShanghaiTech/part_B/train_data/images",
    "dataset/ShanghaiTech/part_B/train_data/ground-truth",
    "dataset/ShanghaiTech/density_maps/part_B"
)

print("Preprocessing complete")