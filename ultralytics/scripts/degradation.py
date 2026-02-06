import cv2
import os
from shutil import copyfile  # 用于复制标签文件
import numpy as np
import random
import math


def calculate_psnr(original, degraded):
    """
    计算原始图像和退化图像之间的PSNR值
    :param original: 原始图像（numpy数组）
    :param degraded: 退化图像（numpy数组）
    :return: PSNR值（dB）
    """
    # 确保图像格式一致
    if original.dtype != degraded.dtype:
        original = original.astype(np.float32)
        degraded = degraded.astype(np.float32)

    # 计算MSE（均方误差）
    mse = np.mean((original - degraded) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同的图像，PSNR无穷大

    # 计算PSNR（假设像素值范围是0-255）
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


# -------------------------- 2. 读取test.txt并提取前200张图片名 --------------------------
def get_top_n_images(test_txt_path, n):
    """从test.txt中读取前n个图片文件名（不含路径，仅文件名）"""
    image_names = []
    with open(test_txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n:
                break  # 只取前n张
            # 去除行尾换行符，确保获取纯净的文件名（如"img_001.jpg"）
            img_name = line.strip()
            if img_name:  # 跳过空行
                image_names.append(img_name)
    # 校验：若test.txt中图片数量不足200，提示警告
    if len(image_names) < n:
        print(f"警告：test.txt中仅包含{len(image_names)}张图片，已全部提取（不足{n}张）")
    return image_names


# -------------------------- 3. 创建新数据集文件夹结构 --------------------------
def create_dataset_dirs(output_root, degradation_params):
    """为每个模糊程度创建YOLO格式的文件夹（images/labels）"""
    for blur_type in degradation_params:
        # 每个模糊程度对应一个数据集文件夹
        blur_dataset_dir = os.path.join(output_root, blur_type)
        # 创建images和labels子文件夹
        degradation_img_dir = os.path.join(blur_dataset_dir, "images")
        degradation_label_dir = os.path.join(blur_dataset_dir, "labels")
        # 递归创建文件夹（若不存在）
        os.makedirs(degradation_img_dir, exist_ok=True)
        os.makedirs(degradation_label_dir, exist_ok=True)
    print(f"已创建所有退化数据集文件夹（共{len(degradation_params)}个程度）")


# -------------------------- 4. 生成模糊图像并迁移标签 --------------------------
def generate_degradation_images_and_labels(
        top_img_paths,
        output_root,
        degradation_params,
        save=False
):
    """
    为每张图片生成多程度模糊图像，并复制对应的YOLO标签
    :param top_img_paths: 前N张图片的文件名列表
    :param degradation_params: 模糊参数字典（key：模糊程度名，value：(kernel, sigma)）
    """
    psnr_stats = {degradation_type: [] for degradation_type in degradation_params.keys()}
    for img_idx, original_img_path in enumerate(top_img_paths, 1):
        img_dir, img_name = os.path.split(original_img_path)
        label_dir, label_name = os.path.join(os.path.dirname(img_dir), 'labels'), os.path.splitext(img_name)[0] + '.txt'
        # 分离文件的目录和文件名
        original_label_path = os.path.join(label_dir, label_name)
        # 校验：跳过不存在的图片/标签（避免报错）
        if not os.path.exists(original_img_path):
            print(f"跳过{img_idx}：原始图片不存在 → {original_img_path}")
            continue
        if not os.path.exists(original_label_path):
            print(f"跳过{img_idx}：标签文件不存在 → {original_label_path}")
            continue

        # 2. 读取原始图片（opencv读取为BGR格式，不影响模糊处理）
        img = cv2.imread(original_img_path)
        if img is None:
            print(f"跳过{img_idx}：图片读取失败 → {original_img_path}")
            continue

        # 3. 为每种模糊程度生成图像并迁移标签
        for degradation_type, param in degradation_params.items():
            if degradation_type.startswith('blur'):
                # 生成模糊图像（cv2.GaussianBlur：高斯模糊核心函数）
                degradation_img = cv2.GaussianBlur(img, (param[0], param[0]), sigmaX=param[1], sigmaY=param[1])
            elif degradation_type.startswith('gaussian'):
                row, col, ch = img.shape
                sigma = param[1]
                gauss = np.random.normal(param[0], sigma, (row, col, ch))
                noisy = img + gauss
                degradation_img = np.clip(noisy, 0, 255).astype(np.uint8)
            elif degradation_type.startswith('fog'):
                color = (255, 255, 255)
                density = param
                distribution = np.ones(img.shape[:2], dtype=np.float32)
                # 基础烟雾
                base_smoke = np.ones(img.shape[:2], dtype=np.float32)
                final_mask = base_smoke * density * distribution
                final_mask = np.clip(final_mask, 0, 1)
                # 雾颜色融合
                smoke_mask_color = np.zeros_like(img, dtype=np.float32)
                for i in range(3):
                    smoke_mask_color[:, :, i] = final_mask * color[i] / 255.0
                image_float = img.astype(np.float32) / 255.0
                smoke_image = image_float * (1 - final_mask[:, :, np.newaxis]) + smoke_mask_color
                degradation_img = np.clip(smoke_image * 255, 0, 255).astype(np.uint8)
            else:
                row, col, ch = img.shape
                noisy = img.copy()

                # 垂直条纹
                num_stripes = int(col * param[1])
                for _ in range(num_stripes):
                    x = random.randint(0, col - 1)
                    stripe_intensity = random.randint(param[0] // 2, param[0])
                    if random.random() > 0.5:
                        noisy[:, x, :] = np.clip(noisy[:, x, :] + stripe_intensity, 0, 255)
                    else:
                        noisy[:, x, :] = np.clip(noisy[:, x, :] - stripe_intensity, 0, 255)

                degradation_img = noisy.astype(np.uint8)

            # 计算PSNR
            psnr = calculate_psnr(img, degradation_img)
            psnr_stats[degradation_type].append(psnr)

            # 构建输出路径
            output_img_dir = os.path.join(output_root, degradation_type, "images")
            output_label_dir = os.path.join(output_root, degradation_type, "labels")
            output_img_path = os.path.join(output_img_dir, img_name)
            output_label_path = os.path.join(output_label_dir, label_name)

            if save:
                # 保存模糊图像
                cv2.imwrite(output_img_path, degradation_img)
                # 复制标签文件（YOLO标签与图像内容无关，直接迁移）
                copyfile(original_label_path, output_label_path)

        # 打印进度（每20张打印一次，避免日志过多）
        if img_idx % 20 == 0 or img_idx == len(top_img_paths):
            print(f"已处理 {img_idx}/{len(top_img_paths)} 张图片")

    # 计算并打印每种退化类型的平均PSNR
    print("\n各退化类型的平均PSNR值：")
    for degradation_type, psnr_values in psnr_stats.items():
        if psnr_values:  # 确保列表不为空
            avg_psnr = sum(psnr_values) / len(psnr_values)
            print(f"- {degradation_type}: 平均PSNR = {avg_psnr:.2f} dB")


def main(d_type, save=False):
    # -------------------------- 1. 配置参数（需根据你的实际路径修改） --------------------------
    TEST_TXT_PATH = r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_degration\test.txt"  # 原始test.txt路径
    OUTPUT_ROOT = r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_degration\tower_gaussian"  # 新数据集根目录（自动创建）
    NUM_IMAGES = 200  # 需提取的前N张图片

    # 高斯模糊参数：(kernel_size, sigma)，kernel_size必须是奇数，sigma越大模糊越严重
    DEGRADATION_PARAMS = {
        "blur": {
            "blur_1": (3, 1),  # 轻度模糊：3x3核，sigma=1
            "blur_2": (7, 3),  # 中度模糊：7x7核，sigma=3
            "blur_3": (15, 7)  # 重度模糊：15x15核，sigma=7
        },
        "gaussian": {
            # 高斯噪声
            "gaussian_noise_1": (0, 10),
            "gaussian_noise_2": (0, 20),
            "gaussian_noise_3": (0, 40),
        },
        "fog": {
            # 雾气效果
            "fog_1": 0.1,
            "fog_2": 0.4,
            "fog_3": 0.7,
        },
        "stripe": {
            "stripe_noise_1": (10, 0.1),
            "stripe_noise_2": (30, 0.2),
            "stripe_noise_3": (60, 0.3)
        }
    }

    # 执行文件夹创建
    create_dataset_dirs(OUTPUT_ROOT, DEGRADATION_PARAMS[d_type].keys())
    # 执行提取
    top_200_imgs = get_top_n_images(TEST_TXT_PATH, NUM_IMAGES)
    print(f"成功提取前{len(top_200_imgs)}张图片，开始生成退化数据集...")
    # 执行核心流程
    generate_degradation_images_and_labels(
        top_200_imgs,
        OUTPUT_ROOT,
        DEGRADATION_PARAMS[d_type],
        save=save
    )

    print("=" * 50)
    print("所有退化数据集生成完成！")
    print(f"新数据集根目录：{os.path.abspath(OUTPUT_ROOT)}")
    for degradation_type, params in DEGRADATION_PARAMS[d_type].items():
        print(
            f"- {degradation_type}: 退化参数:{params}")


if __name__ == '__main__':
    main("gaussian", save=True)
