import random


def select_random_samples(input_file, output_file, sample_size=200):
    """
    从输入文件中随机选择指定数量的行，并保存到输出文件

    参数:
        input_file: 输入文件路径（包含图片路径的test.txt）
        output_file: 输出文件路径（新的测试集文件）
        sample_size: 要选择的样本数量，默认为200
    """
    try:
        # 读取所有图片路径
        with open(input_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # 去除空行
        all_lines = [line.strip() for line in all_lines if line.strip()]

        # 检查文件中的行数是否足够
        if len(all_lines) < sample_size:
            print(f"警告: 输入文件只有{len(all_lines)}行，少于需要选择的{sample_size}行")
            sample_size = len(all_lines)

        # 随机选择指定数量的行
        selected_lines = random.sample(all_lines, sample_size)

        # 将选中的行写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in selected_lines:
                f.write(line + '\n')

        print(f"成功从{len(all_lines)}行中随机选择了{sample_size}行")
        print(f"结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    # 输入文件路径（你的test.txt）
    input_filename = r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower\test.txt"
    # 输出文件路径（新的测试集）
    output_filename = r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_degration\test.txt"
    # 要选择的样本数量
    num_samples = 200

    # 执行选择操作
    select_random_samples(input_filename, output_filename, num_samples)
