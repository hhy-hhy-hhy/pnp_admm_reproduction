"""
PnP-ADMM 图像复原 —— 完整复现代码
支持：自然图像去噪 / MRI 重建
即插即用去噪器：BM3D / OpenCV NLM
"""

import torch
import torch.fft
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==================== 0. 超参数 ====================
class Config:
    # 算法参数
    rho = 0.01              # 惩罚参数初始值
    gamma = 1.1             # 每轮涨幅
    lam = 1e-3              # λ，控制去噪强度 sigma = sqrt(lam/rho)
    tol = 1e-3              # 收敛阈值
    max_iter = 200          # 最大迭代次数
    
    # 去噪器选择：'bm3d' 或 'nlm'
    denoiser_type = 'nlm'   # 默认用 OpenCV NLM，不用装额外库

cfg = Config()

# ==================== 1. 图像加载工具 ====================
def load_image(path, as_tensor=True):
    """读取灰度图，返回 [0,1] 范围的 tensor 或 numpy"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片：{path}")
    img = img.astype(np.float32) / 255.0
    if as_tensor:
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
    return img

def save_image(tensor, path):
    """保存 tensor 为图片"""
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    print(f"💾 已保存：{path}")

# ==================== 2. 去噪器（即插即用） ====================
def denoiser_nlm(img_np, sigma):
    """
    OpenCV 非局部均值去噪（无需安装额外包）
    img_np: (H, W) numpy array, [0,1]
    sigma: 噪声标准差，[0,1]范围
    """
    h = max(0.01, sigma * 255)  # OpenCV 需要 [0,255] 范围的参数
    h = min(h, 30)              # NLM 的 h 不宜过大
    denoised = cv2.fastNlMeansDenoising(
        (img_np * 255).astype(np.uint8), None, h, 7, 21
    )
    return denoised.astype(np.float32) / 255.0

def denoiser_bm3d(img_np, sigma):
    """
    BM3D 去噪（需要 pip install bm3d）
    """
    import bm3d
    denoised = bm3d.bm3d(img_np, sigma_psd=sigma)
    return denoised.astype(np.float32)

def v_update(x_plus_u, sigma):
    """阿净：即插即用去噪"""
    img_np = x_plus_u.squeeze().cpu().numpy()
    sigma_np = sigma  # [0,1] 范围
    
    if cfg.denoiser_type == 'bm3d':
        denoised_np = denoiser_bm3d(img_np, sigma_np)
    else:
        denoised_np = denoiser_nlm(img_np, sigma_np)
    
    return torch.from_numpy(denoised_np).unsqueeze(0)

# ==================== 3. x-更新（核心，按任务不同） ====================
def x_update_denoise(v_minus_u, y, rho):
    """
    自然图像去噪的 x-更新
    前向模型：y = x + noise
    闭式解：x = (y + rho*(v-u)) / (1 + rho)
    """
    return (y + rho * v_minus_u) / (1 + rho)

def get_mri_mask(H, W, sampling_ratio=0.33):
    """
    生成 MRI k-空间欠采样掩膜
    中心全采 + 外围随机射线状采样
    """
    mask = np.zeros((H, W), dtype=np.float32)
    center_size = 16
    cy, cx = H // 2, W // 2
    mask[cy - center_size//2 : cy + center_size//2,
         cx - center_size//2 : cx + center_size//2] = 1.0
    
    # 外围随机采样
    num_lines = int(W * sampling_ratio)
    cols = np.random.choice(W, num_lines, replace=False)
    mask[:, cols] = 1.0
    
    return torch.from_numpy(mask)

def x_update_mri(v_minus_u, y, rho, mask):
    """
    MRI 重建的 x-更新
    在频域用闭式解求解
    """
    # 计算分母：A^T A + rho I 在频域的表示
    denominator = mask + rho  # mask 本身是 0/1
    
    # 计算分子：A^T(y) + rho*(v-u)
    rhs = torch.fft.ifft2(y * mask).real + rho * v_minus_u
    
    # 频域除法
    x = torch.fft.ifft2(torch.fft.fft2(rhs) / denominator.unsqueeze(0)).real
    
    return x

# ==================== 4. 主循环 ====================
def pnp_admm(y, task='denoise', mask=None):
    """
    PnP-ADMM 主循环
    
    参数：
        y: 输入图像 tensor (1, H, W)
        task: 'denoise' 或 'mri'
        mask: MRI 采样掩膜（仅 mri 任务需要）
    
    返回：
        restored: 复原结果 tensor (1, H, W)
        history: dict 包含迭代记录
    """
    H, W = y.shape[-2:]
    
    # 初始化铁三角
    x = y.clone()
    v = y.clone()
    u = torch.zeros_like(y)
    
    rho = cfg.rho
    history = {'psnr': [], 'delta': []}
    
    print(f"\n{'='*50}")
    print(f"🔧 任务：{task.upper()}")
    print(f"📐 图像尺寸：{H}×{W}")
    print(f"🔌 去噪器：{cfg.denoiser_type.upper()}")
    print(f"{'='*50}\n")
    
    for k in range(cfg.max_iter):
        x_old, v_old, u_old = x.clone(), v.clone(), u.clone()
        
        # 4.1 小查画草稿（x-更新）
        if task == 'denoise':
            x = x_update_denoise(v - u, y, rho)
        elif task == 'mri':
            x = x_update_mri(v - u, y, rho, mask)
        else:
            raise ValueError(f"未知任务：{task}")
        
        # 4.2 阿净擦干净（v-更新）
        sigma = np.sqrt(cfg.lam / rho)
        v = v_update(x + u, sigma)
        
        # 4.3 记录员更新偏差（u-更新）
        u = u + (x - v)
        
        # 4.4 自动涨工资（延拓策略）
        rho = rho * cfg.gamma
        
        # 4.5 收敛检查
        with torch.no_grad():
            norm_x = torch.norm(x - x_old).item() / np.sqrt(H * W)
            norm_v = torch.norm(v - v_old).item() / np.sqrt(H * W)
            norm_u = torch.norm(u - u_old).item() / np.sqrt(H * W)
            delta = norm_x + norm_v + norm_u
        
        history['delta'].append(delta)
        
        # 每 5 轮打印一次
        if k % 5 == 0 or delta < cfg.tol:
            print(f"  Iter {k:3d} | δ={delta:.6f} | ρ={rho:.4f} | σ={sigma:.6f}")
        
        if delta < cfg.tol:
            print(f"\n✅ 收敛于第 {k+1} 次迭代，δ={delta:.6f}\n")
            break
    
    if k == cfg.max_iter - 1:
        print(f"\n⚠️ 达到最大迭代次数 {cfg.max_iter}，δ={delta:.6f}\n")
    
    return v, history

# ==================== 5. 评估指标 ====================
def compute_metrics(restored, gt):
    """计算 PSNR 和 SSIM"""
    restored_np = restored.squeeze().detach().cpu().numpy()
    gt_np = gt.squeeze().detach().cpu().numpy()
    
    # 确保范围在 [0,1]
    restored_np = np.clip(restored_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    psnr_val = psnr(gt_np, restored_np, data_range=1.0)
    ssim_val = ssim(gt_np, restored_np, data_range=1.0)
    
    return psnr_val, ssim_val

# ==================== 6. 可视化 ====================
def visualize(gt, noisy, restored, task_name, save_path=None):
    """并排显示：原图 | 损坏图 | 复原图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Ground Truth', 'Degraded', 'PnP-ADMM Restored']
    images = [gt, noisy, restored]
    
    for ax, img, title in zip(axes, images, titles):
        im = img.squeeze().detach().cpu().numpy()
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    plt.suptitle(f'PnP-ADMM: {task_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📊 对比图已保存：{save_path}")
    plt.show()

# ==================== 7. 测试用图像生成 ====================
def create_test_image(size=256):
    """生成一张模拟测试图（如需快速测试用）"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 简单几何图形
    img = np.zeros((size, size), dtype=np.float32)
    img += 0.8 * (np.sqrt(X**2 + Y**2) < 0.6).astype(np.float32)  # 大圆
    img += 0.4 * (np.abs(X - 0.3) + np.abs(Y - 0.3) < 0.2).astype(np.float32)  # 方块
    img += 0.6 * (((X + 0.3)**2 + (Y - 0.2)**2) < 0.1).astype(np.float32)  # 小圆
    
    img = np.clip(img, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)

# ==================== 8. 主程序入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PnP-ADMM 图像复原')
    parser.add_argument('--task', type=str, default='denoise',
                        choices=['denoise', 'mri'],
                        help='任务类型')
    parser.add_argument('--input', type=str, default=None,
                        help='输入图像路径')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground Truth 路径（用于算指标）')
    parser.add_argument('--output', type=str, default='result.png',
                        help='输出图像路径')
    parser.add_argument('--noise_level', type=float, default=0.05,
                        help='高斯噪声标准差（去噪任务）')
    parser.add_argument('--denoiser', type=str, default='nlm',
                        choices=['nlm', 'bm3d'],
                        help='去噪器类型')
    args = parser.parse_args()
    
    cfg.denoiser_type = args.denoiser
    
    # ========== 准备输入图像 ==========
    if args.input and os.path.exists(args.input):
        # 从文件加载
        y = load_image(args.input)
        print(f"📂 加载输入图像：{args.input}")
    else:
        # 创建测试图像
        print("📂 未指定输入，使用内置测试图")
        gt = create_test_image(256)
        noise = torch.randn_like(gt) * args.noise_level
        y = gt + noise
        y = torch.clamp(y, 0, 1)
        args.gt = None  # 用 gt 算指标
        # 把 gt 记下来
        _gt_created = gt.clone()
    
    # ========== 准备 Ground Truth ==========
    gt = None
    if args.gt and os.path.exists(args.gt):
        gt = load_image(args.gt)
    elif 'gt' not in dir() and args.input is None:
        gt = _gt_created
    
    # ========== MRI 掩膜 ==========
    mask = None
    if args.task == 'mri':
        H, W = y.shape[-2:]
        mask = get_mri_mask(H, W, sampling_ratio=0.33)
        # 模拟欠采样：y = IFFT(FFT(gt) * mask)
        if gt is not None:
            y_kspace = torch.fft.fft2(gt) * mask.unsqueeze(0)
            y = torch.fft.ifft2(y_kspace).real
        else:
            # 没有 gt 时，直接对输入做欠采样
            y_kspace = torch.fft.fft2(y) * mask.unsqueeze(0)
            y = torch.fft.ifft2(y_kspace).real
    
    # ========== 运行 PnP-ADMM ==========
    restored, history = pnp_admm(y, task=args.task, mask=mask)
    
    # ========== 评估 ==========
    if gt is not None:
        psnr_val, ssim_val = compute_metrics(restored, gt)
        print(f"\n{'='*50}")
        print(f"📊 评估指标")
        print(f"   PSNR: {psnr_val:.2f} dB")
        print(f"   SSIM: {ssim_val:.4f}")
        print(f"{'='*50}\n")
    
    # ========== 保存结果 ==========
    save_image(restored, args.output)
    
    # ========== 可视化 ==========
    if gt is not None:
        visualize(gt, y, restored, args.task.upper(), 
                  save_path=args.output.replace('.png', '_comparison.png'))
