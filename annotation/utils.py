import os
import shutil
import numpy as np
import cv2
import zipfile
import hashlib
import matplotlib.pyplot as plt


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([0, 0, 0, 0]).astype(np.int64), False
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])
    return np.array([x0, y0, x1, y1]).astype(np.int64), True


def draw_rect(image, bbox, obj_id):
    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id)[:3])
    rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
    inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
    x0, y0, x1, y1 = bbox
    image_with_rect = cv2.rectangle(image.copy(), (x0, y0), (x1, y1), inv_color, thickness=2)
    return image_with_rect


def draw_markers(image, points_dict, labels_dict):
    thickness = 1
    market_relative_size = 0.015

    image_h, image_w = image.shape[:2]
    marker_size = max(1, int(min(image_h, image_w) * market_relative_size))

    for obj_id in points_dict:
        for point, label in zip(points_dict[obj_id], labels_dict[obj_id]):
            x, y = int(point[0]), int(point[1])
            # Green for positive points (label=1), Red for negative points (label=0)
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # BGR format for OpenCV
            if label == 1:
                # Positive points: green cross
                cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=marker_size,
                               thickness=thickness)
            else:
                # Negative points: red tilted cross
                cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=int(marker_size / np.sqrt(2)), thickness=thickness)
    return image


def show_mask(mask, image=None, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0
        for c in range(3):
            image[..., c] = np.where(alpha_mask > 0,
                                     (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c],
                                     image[..., c])
        return image
    return mask_image


def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))


def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
