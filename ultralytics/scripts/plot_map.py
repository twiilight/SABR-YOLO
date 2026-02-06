import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_metrics(csv_files, labels=None, output_file=None):
    # 设置要绘制的指标
    metrics = [
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'val/loss',
    ]

    # 创建5个子图
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    colors = ["#DB7272", "#3C518F", "#38917E", "#C6A550", "#A47DC0", "#4D0085"]

    # 遍历所有指标
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]

        # 遍历所有CSV文件
        for file_idx, csv_file in enumerate(csv_files):
            try:
                df = pd.read_csv(csv_file)

                # 检查列是否存在
                # if metric not in df.columns:
                #     print(f"警告: 文件 {csv_file} 中缺少 '{metric}' 列")
                #     continue

                # 获取标签
                label = labels[file_idx] if labels else Path(csv_file).stem
                data = df['val/box_loss']+df['val/cls_loss']+df['val/dfl_loss'] if metric == 'val/loss' else df[metric]
                # 绘制曲线
                ax.plot(data,
                        label=label,
                        color=colors[file_idx % len(colors)],
                        linewidth=1.5)

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {str(e)}")
                continue

        # 子图装饰
        ax.set_xlabel('Epoch', fontsize=10)
        # ax.set_ylabel(metric.split('/')[-1], fontsize=10)
        ax.set_title(metric.split('/')[-1].split('(')[0], fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        # 只在第一个子图显示图例
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()

    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=800, bbox_inches='tight')
        print(f"图表已保存至 {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    plot_metrics(
        csv_files=[
            r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-raw(n)\results.csv",
            # r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\rsod-yolov3\exp-rsod\results.csv",
            # r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\rsod-yolov8\exp-rsod\results.csv",
            # r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\rsod-yolov10\exp-rsod\results.csv",
            r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-FFCA-YOLO(n)\results.csv",
            r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-SECBAM-BRCFPN(n)6\results.csv"
        ],
        labels=["YOLOv5n", "FFCA-YOLO", "SABR-YOLO"],
        output_file=r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\result\curve_ttod.pdf"
    )
