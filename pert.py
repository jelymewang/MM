import os
import cv2
import numpy as np
import random

# 确保保存路径存在
save_path = "/data_B/renjie/mutimodel"
os.makedirs(save_path, exist_ok=True)

# 图片路径
image_paths = [
    "/data_B/renjie/saw/33/orig.png",
    "/data_B/renjie/saw/61/orig.png"
]

# 定义随机找点并更改颜色的算法
def random_color_points(original_img, num_points=10):
    rows, cols, _ = original_img.shape
    mask = np.ones((rows, cols), dtype=bool)
    zero_pixels = np.all(original_img == 0, axis=2)
    mask[zero_pixels] = False
    non_mask_indices = np.where(mask)
    random_indices = np.random.choice(len(non_mask_indices[0]), size=num_points, replace=False)
    random_points = np.column_stack((non_mask_indices[0][random_indices], non_mask_indices[1][random_indices]))

    target_colors = [
        [0, 0, 30],
        [30, 0, 0],
        [255, 255, 255]
    ]
    image_noise = original_img.copy()
    for index in random_points:
        # 获取当前点的颜色值
        current_color = image_noise[index[0], index[1]]

        # 随机选择目标颜色，确保与当前颜色不一样
        target_color = random.choice([color for color in target_colors if not np.array_equal(color, current_color)])

        # 更改当前点的颜色为目标颜色
        image_noise[index[0], index[1]] = target_color

    return image_noise

# 读取并处理图片
for idx, path in enumerate(image_paths):
    # 读取原始RGB图像
    original_img = cv2.imread(path, -1)
    if original_img is None:
        print(f"无法读取图像：{path}")
        continue
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 保存原始图像
    original_save_filename = os.path.join(save_path, f"original_image_{idx + 1}.jpg")
    cv2.imwrite(original_save_filename, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

    # 添加噪声并保存处理后的图像
    image_with_points = random_color_points(original_img, num_points=30)
    processed_save_filename = os.path.join(save_path, f"processed_image_{idx + 1}.jpg")
    cv2.imwrite(processed_save_filename, cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR))

# 提示保存完成
print(f"原始图像和处理后的图像分别保存至：{save_path}")
