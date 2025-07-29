import cv2
import numpy as np

def crop_square_and_circle(image_path, square_diameter, circle_diameter, output_name, crop_start=(0, 0)):
    """
    1. 将图像裁剪为指定大小的正方形。
    2. 将正方形图像裁剪为圆形。
    3. 从圆形图像中心裁剪出指定直径的圆形。

    Args:
        image_path (str): 输入图像路径。
        square_diameter (int): 要裁剪的正方形的直径。
        circle_diameter (int): 圆形的直径（即从圆形图像中裁剪的圆的直径）。
        output_name (str): 输出文件的名字（用于生成文件路径）。
        crop_start (tuple): 裁剪起点坐标，默认是(0, 0)。
        
    Returns:
        tuple: 返回裁剪后的图像路径元组 (square_output, circle_output, central_circle_output)。
    """
    
    output_path_square = "square_output_" + output_name + ".png"
    output_path_circle = "circle_output_" + output_name + ".png"
    output_path_central_circle = "central_circle_output_" + output_name + ".png"

    # 读取输入图像
    image = cv2.imread(image_path)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 1. 裁剪图像为正方形，裁剪起点为 crop_start
    if height < square_diameter or width < square_diameter:
        print("Error: The image is too small for the desired square crop.")
        return

    # 获取裁剪起点坐标
    x, y = crop_start

    # 检查裁剪范围是否超出图像边界
    if x + square_diameter > width or y + square_diameter > height:
        print("Error: Crop area exceeds image boundaries.")
        return

    # 裁剪图像
    square_image = image[y:y + square_diameter, x:x + square_diameter]

    # 保存裁剪出的正方形图像
    # cv2.imwrite(output_path_square, square_image)

    # 2. 将正方形裁剪为圆形，圆的直径与正方形相同
    # 创建一个黑色掩码
    mask = np.zeros((square_diameter, square_diameter), dtype=np.uint8)

    # 在掩码上绘制白色圆形
    center = (square_diameter // 2, square_diameter // 2)
    radius = square_diameter // 2
    cv2.circle(mask, center, radius, (255), thickness=-1)

    # 将掩码应用到图像上
    circular_image = cv2.bitwise_and(square_image, square_image, mask=mask)

    # 保存圆形图像
    # cv2.imwrite(output_path_circle, circular_image)

    # 3. 从圆形图像中心裁剪出指定直径的圆形
    central_radius = circle_diameter // 2
    if central_radius > radius:
        print("Error: The central circle's radius is larger than the original circle's radius.")
        return

    # 创建中心圆形掩码
    central_mask = np.zeros((square_diameter, square_diameter), dtype=np.uint8)
    cv2.circle(central_mask, center, central_radius, (255), thickness=-1)

    # 从圆形图像中裁剪中心圆形
    central_circular_image = cv2.bitwise_and(circular_image, circular_image, mask=central_mask)

    # 保存中心裁剪出的圆形图像
    cv2.imwrite(output_path_central_circle, central_circular_image)

    return output_path_square, output_path_circle, output_path_central_circle

from PIL import Image

def binarize_image_pil(image_path, threshold=128):
    """
    使用 Pillow (PIL) 对图像进行二值化。

    Args:
        image_path (str): 图像文件路径。
        threshold (int): 二值化阈值，默认为 128。

    Returns:
        PIL.Image.Image: 二值化后的图像，如果读取图像失败则返回 None。
    """
    try:
        # 打开图像
        img = Image.open(image_path).convert("L")  # 转换为灰度图像

        # 二值化
        binarized_img = img.point(lambda x: 0 if x < threshold else 255, mode='1')  # '1' 表示 1-bit 黑白图像
        return binarized_img

    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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

import shutil
import os

def move_files(file_paths, target_folder):
    """
    将文件列表中的文件移动到指定目标文件夹下。

    Args:
        file_paths (list): 包含文件路径的列表。
        target_folder (str): 目标文件夹的路径。
        
    Returns:
        None
    """
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_path in file_paths:
        # 确保文件存在
        if os.path.exists(file_path):
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 构建目标文件路径
            target_path = os.path.join(target_folder, file_name)
            
            # 移动文件
            shutil.move(file_path, target_path)
            print(f"文件 {file_name} 已成功移动到 {target_folder}")
        else:
            print(f"错误: 文件 {file_path} 不存在。")


# # 示例调用

# import time

# start_time = time.time()

# output_dir = './output'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# img = 't10.png'
# path = crop_square_and_circle(img, 250, 220, img[:-4], (60, 75))
# """output = []
# for p in path:
#     output.append(p)
#     print(p)
#     binarized_image = binarize_image_pil(p)
#     op = "binarized_image_pil_" + p
#     output.append(op)
#     print(op)
#     binarized_image.save(op)
#     remove_center_circle(op, 21, "remove_center_"+p)
#     print("remove_center_"+p)
#     output.append("remove_center_"+p)


# move_files(output, output_dir)

# end_time = time.time()
# print(f"运行时间：{end_time - start_time}秒")"""

import time

start_time = time.time()

output_dir = './output_SR'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from moduls.Violet_base import *
paths = list_all_files('output_frames')


for img in paths:
# img = 't10.png'
    
    path = crop_square_and_circle(img, 250, 220, img.split("\\")[-1][:-4], (60, 75))
    output = []
    output.append(path[2])
    move_files(output, output_dir)



end_time = time.time()
print(f"运行时间：{end_time - start_time}秒")
