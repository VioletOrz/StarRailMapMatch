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


# 示例用法：
image_path = "20.png"  # 替换为你的图像路径
binarized_image = binarize_image_pil(image_path)

if binarized_image is not None:
    # 显示二值化后的图像 (可选)
    #binarized_image.show()

    # 保存二值化后的图像 (可选)
    binarized_image.save("binarized_image_pil" + image_path)  # PNG 格式支持 1-bit 图像
else:
    print("Failed to binarize the image.")