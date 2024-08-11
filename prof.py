import torch
from memory_profiler import profile


@profile
def random_color_points(original_img, grad, num_points=10):
    _, _, height, width = original_img.shape
    original_tmp = original_img.squeeze(dim=0).permute((1, 2, 0))
    grad = grad.squeeze(dim=0).permute((1, 2, 0))

    # 创建一个黑色掩码，标记非零像素点
    black = torch.tensor([0, 0, 0]).cuda()
    mask = ~torch.all(original_tmp == black, dim=-1)
    non_zero_indices = torch.nonzero(mask)

    num_non_zero_points = non_zero_indices.size(0)
    num_points = min(int(num_points), num_non_zero_points)

    if num_points == 0:
        return original_img

    # 计算梯度值相加并排序
    grad_masked = grad[mask]
    grad_sum = torch.sum(grad_masked, dim=-1)  # 沿着通道维度进行求和
    sorted_indices = torch.argsort(torch.abs(grad_sum), descending=True)

    modified_indices = 0
    colors = [
        torch.tensor([220, 30, 30]).cuda(),  # 红色
        torch.tensor([30, 30, 200]).cuda(),  # 蓝色
        torch.tensor([255, 255, 255]).cuda()  # 白色
    ]
    color_index = 0

    for index in sorted_indices:
        y, x = non_zero_indices[index]
        pixel = original_tmp[y, x, :]

        # 根据梯度值的正负和通道值的大小选择颜色进行替换
        if grad_sum[index] < 0:
            color_index = 0 if grad[y, x][0] > grad[y, x][2] else 1
        else:
            color_index = 2

        if torch.all(original_tmp[y, x] != colors[color_index]):
            original_tmp[y, x] = colors[color_index]
            modified_indices += 1

        if modified_indices == num_points:
            break

    image_noise = original_tmp.float().permute((2, 0, 1)).unsqueeze(dim=0)

    # 手动删除不再需要的变量
    del grad_masked, grad_sum, original_tmp, sorted_indices, colors, non_zero_indices, mask

    # 释放 PyTorch 使用的缓存空间
    torch.cuda.empty_cache()

    return image_noise


original_img=torch.randn(1, 3, 256, 256).cuda()
grad =torch.randn(1, 3, 256, 256).cuda()
a = random_color_points(original_img,grad,50)
