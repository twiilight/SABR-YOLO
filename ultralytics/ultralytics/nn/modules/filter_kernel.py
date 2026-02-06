import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CannyFilter(nn.Module):
    def __init__(self):
        super(CannyFilter, self).__init__()

        # 高斯滤波器 (5x5)
        self.gaussian_kernel = self._create_gaussian_kernel(5, 1.0)

        # Sobel算子 (3x3)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)

        # 确保卷积核不参与梯度计算
        self.gaussian_kernel.requires_grad = False
        self.sobel_x.requires_grad = False
        self.sobel_y.requires_grad = False

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯滤波器"""
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        kernel /= kernel.sum()  # 归一化
        return kernel.unsqueeze(0).unsqueeze(0)  # 增加批次和通道维度

    def forward(self, x, low_threshold=0.1, high_threshold=0.3):
        """
        对多通道图像的每个通道执行Canny边缘检测

        参数:
            x: 输入图像，形状为 (B, C, H, W)
            low_threshold: 低阈值，用于双阈值处理
            high_threshold: 高阈值，用于双阈值处理

        返回:
            edges: 每个通道的边缘检测结果，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        edges = torch.zeros_like(x)

        # 对每个通道独立执行Canny边缘检测
        for c in range(C):
            # 提取单个通道
            single_channel = x[:, c:c + 1, :, :]

            # 步骤1: 高斯滤波降噪
            x_blur = F.conv2d(single_channel, self.gaussian_kernel, padding=self.gaussian_kernel.size(-1) // 2)

            # 步骤2: 计算梯度幅值和方向
            grad_x = F.conv2d(x_blur, self.sobel_x, padding=1)
            grad_y = F.conv2d(x_blur, self.sobel_y, padding=1)

            # 计算梯度幅值
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            # 归一化梯度幅值到[0, 1]范围
            grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

            # 计算梯度方向 (弧度)
            grad_dir = torch.atan2(grad_y, grad_x)
            # 转换为角度并调整到0-180度
            grad_dir = torch.rad2deg(grad_dir)
            grad_dir += 180.0
            grad_dir %= 180.0

            # 步骤3: 非极大值抑制
            nms_mag = self._non_max_suppression(grad_mag, grad_dir)

            # 步骤4: 双阈值处理和边缘连接
            channel_edges = self._double_threshold(nms_mag, low_threshold, high_threshold)

            # 将结果存入对应通道
            edges[:, c:c + 1, :, :] = channel_edges

        return edges

    def _non_max_suppression(self, grad_mag, grad_dir):
        """执行非极大值抑制"""
        B, C, H, W = grad_mag.shape
        nms_mag = torch.zeros_like(grad_mag)

        # 遍历每个像素
        for b in range(B):
            for c in range(C):
                for i in range(1, H - 1):
                    for j in range(1, W - 1):
                        angle = grad_dir[b, c, i, j]
                        mag = grad_mag[b, c, i, j]

                        # 确定梯度方向的四个可能象限
                        if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                            # 水平方向 (0度)
                            neighbor1 = grad_mag[b, c, i, j + 1]
                            neighbor2 = grad_mag[b, c, i, j - 1]
                        elif 22.5 <= angle < 67.5:
                            # 45度方向
                            neighbor1 = grad_mag[b, c, i + 1, j - 1]
                            neighbor2 = grad_mag[b, c, i - 1, j + 1]
                        elif 67.5 <= angle < 112.5:
                            # 垂直方向 (90度)
                            neighbor1 = grad_mag[b, c, i + 1, j]
                            neighbor2 = grad_mag[b, c, i - 1, j]
                        else:  # 112.5 <= angle < 157.5
                            # 135度方向
                            neighbor1 = grad_mag[b, c, i + 1, j + 1]
                            neighbor2 = grad_mag[b, c, i - 1, j - 1]

                        # 只有当当前像素的梯度幅值大于两个邻居时才保留
                        if mag >= neighbor1 and mag >= neighbor2:
                            nms_mag[b, c, i, j] = mag

        return nms_mag

    def _double_threshold(self, nms_mag, low_threshold, high_threshold):
        """双阈值处理和边缘连接"""
        B, C, H, W = nms_mag.shape
        edges = torch.zeros_like(nms_mag)

        # 高阈值标记为强边缘
        strong_edges = (nms_mag >= high_threshold).float()
        # 低阈值到高阈值之间标记为弱边缘
        weak_edges = (nms_mag >= low_threshold) & (nms_mag < high_threshold)

        edges = strong_edges.clone()

        # 连接弱边缘到强边缘
        for b in range(B):
            for c in range(C):
                for i in range(1, H - 1):
                    for j in range(1, W - 1):
                        if weak_edges[b, c, i, j]:
                            # 检查3x3邻域内是否有强边缘
                            if torch.any(strong_edges[b, c, i - 1:i + 2, j - 1:j + 2]):
                                edges[b, c, i, j] = 1.0

        return edges


# 使用示例
if __name__ == "__main__":

    # 创建多通道Canny边缘检测器实例
    canny = MultiChannelCannyEdgeDetector()

    # 创建一个多通道示例图像 (批次大小=2, 通道=3, 高度=256, 宽度=256)
    example_image = torch.rand(2, 3, 256, 256)

    # 运行Canny边缘检测
    with torch.no_grad():  # 推理时不需要计算梯度
        edges = canny(example_image, low_threshold=0.1, high_threshold=0.3)

    print(f"输入图像形状: {example_image.shape}")
    print(f"边缘检测结果形状: {edges.shape}")
    print(f"边缘检测结果范围: [{edges.min():.4f}, {edges.max():.4f}]")
