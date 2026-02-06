import os


def find_multi_instance_images(images_dir, labels_dir):
    """
    找出YOLO格式数据集中包含多个instance的图片

    参数:
        images_dir: 存放图片的目录路径
        labels_dir: 存放对应标注文件的目录路径
    """
    multi_instance_images = []

    # 遍历标签目录中的所有文件
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)

            # 统计当前文件中的instance数量
            with open(label_path, 'r') as f:
                lines = f.readlines()
                instance_count = len([line for line in lines if line.strip()])

            # 如果instance数量大于1，则记录对应的图片
            if instance_count > 1:
                # 构建对应的图片文件名（假设图片和标签文件名相同，只是扩展名不同）
                image_name = os.path.splitext(label_file)[0] + '.jpg'  # 根据实际情况调整图片扩展名
                if os.path.exists(os.path.join(images_dir, image_name)):
                    multi_instance_images.append((image_name, instance_count))
                else:
                    print(f"警告: 找不到图片 {image_name} 对应的标签文件 {label_file}")

    return multi_instance_images


if __name__ == "__main__":
    # 设置你的图片和标签目录路径
    images_directory = r"D:\qzx\tower_dataset\tower\images"  # 替换为你的图片目录路径
    labels_directory = r"D:\qzx\tower_dataset\tower\labels"  # 替换为你的标签目录路径

    # 查找包含多个instance的图片
    results = find_multi_instance_images(images_directory, labels_directory)

    # 打印结果
    if results:
        print(f"找到 {len(results)} 张包含多个instance的图片:")
        for image, count in results:
            print(f"{image}: {count}个instance")
    else:
        print("没有找到包含多个instance的图片")