import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import os

# ========== 0. 你的数据文件列表 ==========
file_list = [
    'data/test_data_01.npy',
    'data/test_data_02.npy',
    'data/test_data_03.npy',
]

# ========== 1. 通用 MRI 掩膜生成 ==========
def make_mask(H, W, center_size=16, sampling_ratio=0.4):
    mask = torch.zeros(H, W)
    cy, cx = H // 2, W // 2
    mask[cy - center_size//2 : cy + center_size//2,
         cx - center_size//2 : cx + center_size//2] = 1.0
    n_lines = max(1, int(W * sampling_ratio))
    cols = torch.randperm(W)[:n_lines]
    mask[:, cols] = 1.0
    return mask.float()

# ========== 2. 去噪器 (OpenCV NLM) ==========
def denoiser(img_np, sigma):
    h = max(0.1, sigma * 255)
    h = min(h, 30)
    den = cv2.fastNlMeansDenoising((img_np * 255).astype(np.uint8), None, h, 7, 21)
    return den.astype(np.float32) / 255.0

# ========== 3. 单个文件的处理流程 ==========
def process_one_file(filepath, output_dir='data'):
    print(f"\n{'='*60}")
    print(f"正在处理：{filepath}")

    # --- 3.1 加载数据 ---
    arr = np.load(filepath, allow_pickle=True)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # 如果是3D，取中间切片
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    # 取绝对值（MRI幅值），归一化到 [0, 1]
    arr = np.abs(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    gt_arr = arr.astype(np.float32)

    # --- 3.2 转为 Ground Truth Tensor ---
    gt = torch.from_numpy(gt_arr).unsqueeze(0)  # (1, H, W)
    H, W = gt.shape[-2:]

    # --- 3.3 生成掩膜并模拟欠采样 k 空间 ---
    mask = make_mask(H, W, center_size=16, sampling_ratio=0.4)
    kspace_full = torch.fft.fft2(gt)
    kspace_obs = kspace_full * mask.unsqueeze(0)   # 欠采样的 k 空间
    zero_filled = torch.fft.ifft2(kspace_obs).real  # 零填充重建（作为输入对比）

    # --- 3.4 PnP-ADMM 参数设置 ---
    rho = 0.01
    gamma = 1.1
    lam = 1e-3
    max_iter = 200
    tol = 1e-3

    # 初始化
    x = zero_filled.clone()
    v = x.clone()
    u = torch.zeros_like(x)

    # --- 3.5 主循环 ---
    print("开始 MRI 重建...")
    for k in range(max_iter):
        x_old, v_old, u_old = x.clone(), v.clone(), u.clone()

        # x-更新：频域闭式解
        sigma = np.sqrt(lam / rho)
        numerator = kspace_obs + rho * torch.fft.fft2(v - u)
        denominator = mask.unsqueeze(0) + rho
        x = torch.fft.ifft2(numerator / denominator).real

        # v-更新：去噪器
        v_np = (x + u).squeeze().cpu().numpy()
        v_np = np.clip(v_np, 0, 1)
        v_denoised = denoiser(v_np, sigma)
        v = torch.from_numpy(v_denoised).unsqueeze(0).float()

        # u-更新
        u = u + (x - v)

        # ρ 延拓
        rho = rho * gamma

        # 收敛检查
        delta = (torch.norm(x - x_old) + torch.norm(v - v_old) + torch.norm(u - u_old)) / np.sqrt(H * W)
        if k % 10 == 0 or delta < tol:
            print(f"Iter {k:3d}  δ={delta:.6f}  ρ={rho:.4f}  σ={sigma:.6f}")
        if delta < tol:
            print(f"✅ 收敛于第 {k+1} 次迭代，δ={delta:.6f}")
            break

    # --- 3.6 结果评估 ---
    restored = v  # v 通常更好
    restored_np = restored.squeeze().cpu().numpy()
    psnr_val = psnr(gt_arr, restored_np, data_range=1.0)
    ssim_val = ssim(gt_arr, restored_np, data_range=1.0)
    print(f"\n📊 {os.path.basename(filepath)}")
    print(f"   PSNR: {psnr_val:.2f} dB")
    print(f"   SSIM: {ssim_val:.4f}")

    # --- 3.7 保存结果 ---
    base = os.path.splitext(os.path.basename(filepath))[0]
    recon_path = os.path.join(output_dir, f"{base}_mri_recon.png")
    comp_path = os.path.join(output_dir, f"{base}_mri_compare.png")

    plt.imsave(recon_path, restored_np, cmap='gray')
    print(f"💾 重建结果已保存：{recon_path}")

    # 三栏对比图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_arr, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[1].imshow(zero_filled.squeeze().numpy(), cmap='gray')
    axes[1].set_title('Zero‑filled (Input)')
    axes[2].imshow(restored_np, cmap='gray')
    axes[2].set_title('PnP‑ADMM')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150)
    print(f"💾 对比图已保存：{comp_path}")

    return psnr_val, ssim_val

# ========== 4. 批量处理 ==========
if __name__ == '__main__':
    for f in file_list:
        process_one_file(f)
    print("\n🎉 所有 MRI 数据重建完成！")
