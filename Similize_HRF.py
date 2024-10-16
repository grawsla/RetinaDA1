import numpy as np
import os
import random
import cv2
from PIL import Image
import hashlib

# 控制执行的次数
execution_count = 1
current_index = 4
photo_index = 1

# GIF/TIF 转 PNG 函数
def convert_to_png(image_path, output_dir):
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 修改为不包含后缀的文件名
    file_name = base_name
    
    # 保存为 PNG 格式
    png_path = os.path.join(output_dir, f"{file_name}.png")
    img.save(png_path, 'PNG')
    print(f"Converted {os.path.splitext(os.path.basename(image_path))[1]} to PNG: {png_path}")
        
    return png_path

# 判断是否为 GIF 或 TIF 格式
def is_convertible(image_path):
    return image_path.lower().endswith(('.gif', '.tif'))

# 图像处理函数
def process_image(image_path, output_dir, center, seed, size_range):
    random.seed(seed)  # 初始化随机数生成器
    # 0.预处理
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 将图像加载为灰度图
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(image)

    # 1. 加载 PNG 图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 使用cv2.IMREAD_COLOR参数
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 3. 添加旋转操作
    photo_angle = random.randint(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), photo_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # 随机选择是否翻转
    photo_fz = random.randint(0, 1)
    if photo_fz:
        flipped_image = cv2.flip(rotated_image, 0)  # 沿着垂直轴翻转
    else:
        flipped_image = rotated_image

    # 4. 基于坐标的裁剪
    center_x, center_y = center
    size = random.randint(size_range[0], size_range[1])  # 随机裁剪的大小
    center_x = center_x + random.randint(-150, 150)  # 随机偏移裁剪中心点
    center_y = center_y + random.randint(-150, 150)  # 随机偏移裁剪中心点

    start_x = max(min(center_x - size // 2, image.shape[1] - size), 0)
    start_y = max(min(center_y - size // 2, image.shape[0] - size), 0)
    end_x = min(center_x + size // 2, image.shape[1])
    end_y = min(center_y + size // 2, image.shape[0])

    cropped_image = flipped_image[start_y:end_y, start_x:end_x]

    # 数据标准化
    cropped_image = ((cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image)) * 255).astype(np.uint8)
    resized_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)

    # 保存裁剪后的图像
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 修改为不包含后缀的文件名
    output_path = os.path.join(output_dir, f"{base_name}_{photo_index}.png")
    Image.fromarray(resized_image).save(output_path)
    print(f"Cropped image saved to: {output_path}")

# 输入文件夹路径
input_dir1 = 'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\training\\images'
input_dir2 = 'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\training\\manual1'

# 初始输出文件夹路径
output_dir1 = f'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\output\\images_{current_index}'
output_dir2 = f'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\output\\manual1_{current_index}'

# 初始裁剪大小范围
size_range = [1000, 1100]

# 确保输出目录存在
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

# 获取两个文件夹中的文件名（不包含后缀）
def get_file_name_without_extension(file_name):
    return os.path.splitext(file_name)[0]

# 构建文件名对应的文件名字典
files1 = {get_file_name_without_extension(f): f for f in os.listdir(input_dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.gif'))}
files2 = {get_file_name_without_extension(f): f for f in os.listdir(input_dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.gif'))}

# 确保两个文件夹中的文件名相同（不包含后缀）
common_files = set(files1.keys()).intersection(files2.keys())

# 存储中心坐标和随机种子
centers = {}

for file_name in common_files:
    image_path1 = os.path.join(input_dir1, files1[file_name])
    
    # 如果是 GIF 或 TIF 文件，先转换为 PNG
    if is_convertible(image_path1):
        image_path1 = convert_to_png(image_path1, output_dir1)

    image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(image)
    x, y = max_loc[0], max_loc[1]
    centers[file_name] = (x, y)

for iteration in range(execution_count):
    for file_name in common_files:
        seed = hash(file_name)  # 使用文件名的哈希值作为种子
        image_path1 = os.path.join(input_dir1, files1[file_name])
        image_path2 = os.path.join(input_dir2, files2[file_name])
        
        # 如果是 GIF 或 TIF 文件，先转换为 PNG
        if is_convertible(image_path1):
            image_path1 = convert_to_png(image_path1, output_dir1)
        if is_convertible(image_path2):
            image_path2 = convert_to_png(image_path2, output_dir2)

        center = centers[file_name]
        for i in range(20):
            center = centers[file_name]
            seed = hash(file_name) + i  # 增加随机数种子
            process_image(image_path1, output_dir1, center, seed, size_range)
            process_image(image_path2, output_dir2, center, seed, size_range)
            photo_index += 1
        photo_index = 1  # 重置 photo_index
    # 更新输出目录
    current_index += 1
    output_dir1 = f'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\output\\images_{current_index}'
    output_dir2 = f'D:\\Study\\Project1\\Data\\dataset collection\\HRF\\output\\manual1_{current_index}'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    
    # 更新裁剪大小范围
    size_range[0] += 50
    size_range[1] += 50