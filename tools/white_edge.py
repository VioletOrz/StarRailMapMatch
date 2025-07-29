import cv2
import numpy as np
from PIL import Image  # 导入 Pillow 库


def detect_white_edges_enhanced(image_path, lower_threshold=180, upper_threshold=255, blur_size=(5, 5), min_area=20):
    """
    检测图像中的白色（或接近白色）边缘线，并输出边缘线的图像。 增强版本，减少噪点影响。

    Args:
        image_path (str): 图像文件路径。
        lower_threshold (int): 像素值下限，用于判断是否接近白色。
        upper_threshold (int): 像素值上限，用于判断是否接近白色。
        blur_size (tuple): 高斯模糊的内核大小，例如 (5, 5)。
        min_area(int): 最小面积，小于该面积的轮廓会被忽略

    Returns:
        numpy.ndarray: 边缘线的图像（白色边缘，黑色背景），如果读取图像失败则返回 None。
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 模糊处理 (减少噪点)
        blurred = cv2.GaussianBlur(gray, blur_size, 0)

        # 使用阈值分割，提取接近白色的区域，使用范围阈值
        _, white_mask = cv2.threshold(blurred, lower_threshold, upper_threshold, cv2.THRESH_BINARY)

        # 形态学操作 (去除噪点和填充小孔)
        kernel = np.ones((3, 3), np.uint8)  # 定义一个 3x3 的内核
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算去除小噪点
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # 闭运算填充小孔

        # 使用 Canny 边缘检测
        edges = cv2.Canny(white_mask, 50, 150)

        # 创建一个黑色图像作为背景
        edge_image = np.zeros_like(gray)

        # 将检测到的边缘设置为白色
        edge_image[edges == 255] = 255

        # 轮廓检测， 过滤小面积轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个黑色图像作为背景
        edge_image = np.zeros_like(gray)
        # 筛选掉小轮廓
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(edge_image, [contour], -1, 255, 1) # 绘制轮廓
        return edge_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# 示例用法
image_path = "imt3.png"  # 替换为你的图像路径
edge_image = detect_white_edges_enhanced(image_path, lower_threshold=190, upper_threshold=255, blur_size=(7, 7), min_area=30)

if edge_image is not None:
    # 使用 Pillow 显示图像
    img_pil = Image.fromarray(edge_image)  # 将 NumPy 数组转换为 Pillow 图像
    img_pil.show()  # 显示图像

    # 保存图像
    cv2.imwrite("white_edges_enhanced4.jpg", edge_image)
else:
    print("Failed to detect white edges.")