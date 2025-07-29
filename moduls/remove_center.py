import cv2
import numpy as np

def remove_center_circle(image_path, radius, output_path="output.png"):
    """
    去除图像中心指定半径的圆形区域，改为黑色。

    Args:
        image_path (str): 输入图像路径。
        radius (int): 要去除的圆形区域半径（像素）。
        output_path (str): 处理后图像的保存路径。

    Returns:
        None
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 获取图像中心坐标
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 创建一个黑色掩码
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # 在掩码上绘制黑色圆形
    cv2.circle(mask, center, radius, 0, thickness=-1)

    # 如果图像有 alpha 通道，需单独处理
    if image.shape[-1] == 4:  # RGBA 图像
        image[:, :, 3] = cv2.bitwise_and(image[:, :, 3], mask)
    else:  # RGB/BGR 图像
        image = cv2.bitwise_and(image, image, mask=mask)

    # 保存处理后的图像
    cv2.imwrite(output_path, image)

# 示例调用
img = '20.png'
remove_center_circle("binarized_image_pil"+img, 21, "remove_center_"+img)



