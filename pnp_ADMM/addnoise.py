import cv2
import numpy as np
import os

# 设置你的干净图片路径和想要的噪声强度
clean_path = 'data/06.png'      # 你的干净原图
noisy_path = 'data/my_noisy06.png'      # 要生成的带噪图
noise_level = 0.05                    # 噪声强度，0.03是轻微，0.1是很重

# 读取干净图像，像素值缩放到 [0, 1]
img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

# 加上高斯噪声
noise = np.random.randn(*img.shape) * noise_level
noisy_img = img + noise

# 让像素值乖乖待在 [0, 1] 范围内
noisy_img = np.clip(noisy_img, 0, 1)

# 保存为图片
cv2.imwrite(noisy_path, (noisy_img * 255).astype(np.uint8))
print(f'✅ 带噪图已生成：{noisy_path}')
