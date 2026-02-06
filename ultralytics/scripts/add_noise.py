import os

import numpy as np
import cv2
import math


def add_noise(src, dst):
    src_img = cv2.imread(src)
    h, w, c = src_img.shape

    dst_img = cv2.blur(src_img, (5, 5))
    cv2.imwrite(os.path.join(dst, "blur.jpg"), dst_img)

    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, src_img.shape)
    noisy_img = src_img + gauss
    dst_img = np.clip(noisy_img, a_min=0, a_max=255)
    cv2.imwrite(os.path.join(dst, "gauss.jpg"), dst_img)

    dst_img = src_img.copy()
    j = 0
    while j < w:
        if np.random.random() < 0.05:
            intensity = np.random.uniform(0.1, 0.8)
            noise = np.ones((h, c)) * intensity * 255.0
            # noise = np.zeros((h, c), dtype=np.uint8)
            dst_img[:, j, :] = noise
        j += 1
    cv2.imwrite(os.path.join(dst, "stripe.jpg"), dst_img)

    color = (255, 255, 255)
    density = 0.2
    distribution = np.ones((h, w), dtype=np.float32)
    # 基础烟雾
    base_smoke = np.ones((h, w), dtype=np.float32)
    final_mask = base_smoke * density * distribution
    final_mask = np.clip(final_mask, 0, 1)
    # 雾颜色融合
    smoke_mask_color = np.zeros_like(src_img, dtype=np.float32)
    for i in range(3):
        smoke_mask_color[:, :, i] = final_mask * color[i] / 255.0
    image_float = src_img.astype(np.float32) / 255.0
    smoke_image = image_float * (1 - final_mask[:, :, np.newaxis]) + smoke_mask_color
    dst_img = np.clip(smoke_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(dst, "fog.jpg"), dst_img)


if __name__ == '__main__':
    add_noise(r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_noise_2\107982_3.jpg",
              r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_noise_2")
