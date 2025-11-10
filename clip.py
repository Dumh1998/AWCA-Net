import numpy as np
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
from skimage import io
import os

import numpy as np


def truncated_linear_stretch(
        image, truncated_percent=2, stretch_range=[0, 255], is_drop_non_positive=False
):
    """_summary_

    Args:
        image (np.array): HWC or HW
        truncated_percent (int, optional): _description_. Defaults to 2.
        stretch_range (list, optional): _description_. Defaults to [0, 255].
    """
    max_tansformed_img = (
        np.where(image <= 0, 65536, image) if is_drop_non_positive else image
    )
    min_tansformed_img = (
        np.where(image <= 0, -65536, image) if is_drop_non_positive else image
    )

    truncated_lower = np.percentile(
        max_tansformed_img, truncated_percent, axis=(0, 1), keepdims=True
    )
    truncated_upper = np.percentile(
        min_tansformed_img, 100 - truncated_percent, axis=(0, 1), keepdims=True
    )

    stretched_img = (image - truncated_lower) / (truncated_upper - truncated_lower) * (
            stretch_range[1] - stretch_range[0]
    ) + stretch_range[0]
    stretched_img[stretched_img < stretch_range[0]] = stretch_range[0]
    stretched_img[stretched_img > stretch_range[1]] = stretch_range[1]
    if stretch_range[1] <= 255:
        stretched_img = np.uint8(stretched_img)
    elif stretch_range[1] <= 65535:
        stretched_img = np.uint16(stretched_img)
    return stretched_img


save_dir_path = Path("xxx/tif")
save_vis_dir_path = Path("xxx/vis")
save_dir_path.mkdir(parents=True, exist_ok=True)
save_vis_dir_path.mkdir(parents=True, exist_ok=True)

img_dir_path = Path("xxx")

img_pre_dir_path = img_dir_path / "A"
img_post_dir_path = img_dir_path / "B"
img_label_dir_path = img_dir_path / "label"

img_name_list = [img_path.name for img_path in img_pre_dir_path.glob("*.tif")]

patch_size = 256
stride = patch_size // 2

for img_name in tqdm(img_name_list):
    img_pre_path = img_pre_dir_path / img_name
    img_post_path = img_post_dir_path / img_name
    img_label_path = img_label_dir_path / img_name

    # Load images
    img_pre = tiff.imread(img_pre_path)
    img_post = tiff.imread(img_post_path)
    img_label = tiff.imread(img_label_path)

    h, w = img_pre.shape[:2]

    patch_h_num = (h - patch_size) // stride + 1
    patch_w_num = (w - patch_size) // stride + 1

    for h_idx in range(patch_h_num):
        for w_idx in range(patch_w_num):
            h_start = h_idx * stride
            h_end = h_start + patch_size
            w_start = w_idx * stride
            w_end = w_start + patch_size

            if h_end > h:
                h_end = h
                h_start = h_end - patch_size
            if w_end > w:
                w_end = w
                w_start = w_end - patch_size

            img_pre_patch = img_pre[h_start:h_end, w_start:w_end]
            img_post_patch = img_post[h_start:h_end, w_start:w_end]
            img_label_patch = img_label[h_start:h_end, w_start:w_end]

            vis_img_pre_patch = truncated_linear_stretch(img_pre_patch)
            vis_img_post_patch = truncated_linear_stretch(img_post_patch)
            base_name = os.path.splitext(img_name)[0]
            img_pre_patch_save_path = save_dir_path / "A" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.tif"
            img_post_patch_save_path = save_dir_path / "B" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.tif"
            img_label_patch_save_path = save_dir_path / "label" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.tif"

            vis_img_pre_patch_save_path = save_vis_dir_path / "A" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.png"
            vis_img_post_patch_save_path = save_vis_dir_path / "B" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.png"
            vis_img_label_patch_save_path = save_vis_dir_path / "label" / f"{base_name}_{h_start:05d}_{h_end:05d}_{w_start:05d}_{w_end:05d}.png"

            img_pre_patch_save_path.parent.mkdir(parents=True, exist_ok=True)
            img_post_patch_save_path.parent.mkdir(parents=True, exist_ok=True)
            img_label_patch_save_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img_pre_patch_save_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img_post_patch_save_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img_label_patch_save_path.parent.mkdir(parents=True, exist_ok=True)

            tiff.imwrite(img_pre_patch_save_path, img_pre_patch)
            tiff.imwrite(img_post_patch_save_path, img_post_patch)
            tiff.imwrite(img_label_patch_save_path, img_label_patch)
            io.imsave(vis_img_pre_patch_save_path, vis_img_pre_patch)
            io.imsave(vis_img_post_patch_save_path, vis_img_post_patch)
            io.imsave(vis_img_label_patch_save_path, img_label_patch)


