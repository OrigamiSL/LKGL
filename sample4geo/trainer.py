import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
import os
import argparse
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
from PIL import Image
import cv2
import json

def load_first_tensor(pth_path: str):
    data = torch.load(pth_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        t = data
    elif isinstance(data, dict):
        tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            raise ValueError(f"No tensor found in {pth_path}")
        t = tensors[0]
    else:
        raise ValueError(f"Unsupported .pth content in {pth_path}")
    t = t.squeeze().cpu()
    # if there is a leading batch dimension, take first item
    if t.ndim > 3:
        t = t[0]
    return t


def ensure_grid_tensor(tensor: torch.Tensor, n: int):
    """
    Ensure tensor becomes shape (n, n, d). Accepts:
      - already (n, n, d)
      - (n, n) -> treat d=1
      - any shape with total elements divisible by n*n -> reshape to (n, n, d)
    Returns numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        t = tensor
    else:
        t = torch.tensor(tensor)
    if t.ndim == 2 and t.shape[0] == n and t.shape[1] == n:
        return t.numpy()[..., None]
    if t.ndim == 3 and t.shape[0] == n and t.shape[1] == n:
        return t.numpy()
    total = t.numel()
    if total % (n * n) == 0:
        d = total // (n * n)
        return t.reshape(n, n, d).numpy()
    raise ValueError(f"Cannot reshape tensor of shape {tuple(t.shape)} to ({n},{n},*)")


def tsne_reduce(arr3d, n=12, seed=0):
    """
    Reduce last-dim vectors to scalar via TSNE (n_components=1).
    Returns numpy array shape (n,n) of scalars.
    """
    arr = ensure_grid_tensor(arr3d, n)
    X = arr.reshape(-1, arr.shape[-1])  # (n*n, d)
    tsne = TSNE(
        n_components=1,
        init="pca",
        random_state=seed,
        perplexity=30,
        learning_rate="auto",
    )
    y = tsne.fit_transform(X).reshape(n, n)
    return y


def plot_surface_grid_save(grid, save_path, cmap="magma", dpi=600, figsize=(6, 6)):
    """
    Plot 3D surface and save to disk (transparent background).
    This keeps the original look: axis off, default view.
    """
    grid = np.asarray(grid, dtype=float)

    # normalize (for color only)
    gmin, gmax = grid.min(), grid.max()
    if gmax > gmin:
        norm = (grid - gmin) / (gmax - gmin)
    else:
        norm = np.zeros_like(grid)

    n = grid.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        grid,
        facecolors=plt.get_cmap(cmap)(norm),
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # clean look
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)

    # default view
    ax.view_init(elev=30, azim=-60)

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)


def render_surface_to_image(grid, cmap="magma", dpi=600, figsize=(6, 6)):
    """
    Render the 3D surface into a PIL.Image (RGBA) saved into memory.
    """
    grid = np.asarray(grid, dtype=float)
    gmin, gmax = grid.min(), grid.max()
    if gmax > gmin:
        norm = (grid - gmin) / (gmax - gmin)
    else:
        norm = np.zeros_like(grid)

    n = grid.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        grid,
        facecolors=plt.get_cmap(cmap)(norm),
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    ax.view_init(elev=30, azim=-60)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    buf.close()
    return img


def render_heatmap_to_image(grid, cmap="magma", alpha=1.0, dpi=600, figsize=(6, 6), remove_axes=True):
    """
    Render 2D heatmap with the same vmin/vmax normalization as used for the surface,
    no colorbar, no ticks. Returns PIL.Image (RGBA).
    """
    grid = np.asarray(grid, dtype=float)
    gmin, gmax = grid.min(), grid.max()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # show heatmap with explicit vmin/vmax so it aligns with surface colors
    im = ax.imshow(grid, cmap=cmap, vmin=gmin, vmax=gmax, origin="lower", interpolation="nearest")

    if remove_axes:
        ax.set_axis_off()
    else:
        plt.colorbar(im)

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    buf.close()

    # apply uniform alpha scalar to entire heatmap image (preserves per-pixel alpha but scales it)
    if alpha is not None and alpha < 1.0:
        arr = np.array(img)
        # scale existing alpha channel by alpha factor
        arr[..., 3] = (arr[..., 3].astype(np.float32) * float(alpha)).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGBA")

    return img

import numpy as np
from PIL import Image

# def composite_visible_and_heatmap(
#     visible_np: np.ndarray,
#     heatmap,
#     heat_alpha: float = 0.5,
#     resize_method=Image.BILINEAR,
#     output_numpy: bool = False,
# ):
#     """
#     Overlay heatmap on a visible image.

#     Supports visible image in:
#       - CHW (3, H, W)   ← your case
#       - HWC (H, W, 3)
#       - uint8 or float

#     Returns:
#       - PIL.Image (RGB) or numpy.ndarray (H, W, 3)
#     """

#     # ---------- FIX 1: CHW -> HWC ----------
#     if visible_np.ndim == 3 and visible_np.shape[0] in (1, 3, 4):
#         # assume CHW
#         visible_np = np.transpose(visible_np, (1, 2, 0))

#     # ---------- FIX 2: dtype ----------
#     if visible_np.dtype != np.uint8:
#         visible_np = np.clip(visible_np, 0, 255).astype(np.uint8)

#     # ---------- FIX 3: channels ----------
#     if visible_np.ndim == 2:
#         visible_np = np.stack([visible_np] * 3, axis=-1)

#     if visible_np.shape[-1] == 3:
#         alpha = 255 * np.ones((*visible_np.shape[:2], 1), dtype=np.uint8)
#         visible_rgba = np.concatenate([visible_np, alpha], axis=-1)
#     else:
#         visible_rgba = visible_np

#     vis = Image.fromarray(visible_rgba, mode="RGBA")

#     # ---------- heatmap ----------
#     if isinstance(heatmap, np.ndarray):
#         if heatmap.dtype != np.uint8:
#             heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
#         heat = Image.fromarray(heatmap)
#     else:
#         heat = heatmap

#     heat = heat.convert("RGBA")

#     if heat.size != vis.size:
#         heat = heat.resize(vis.size, resample=resize_method)

#     # ---------- alpha blending ----------
#     vis_arr = np.asarray(vis).astype(np.float32) / 255.0
#     heat_arr = np.asarray(heat).astype(np.float32) / 255.0

#     a_heat = heat_arr[..., 3:4] * float(np.clip(heat_alpha, 0.0, 1.0))

#     out_rgb = heat_arr[..., :3] * a_heat + vis_arr[..., :3] * (1.0 - a_heat)
#     out = (np.clip(out_rgb, 0.0, 1.0) * 255).astype(np.uint8)

#     if output_numpy:
#         return out  # (H, W, 3)

#     return Image.fromarray(out, mode="RGB")
import numpy as np
import cv2
from typing import Tuple, Union

# colormap mapping (支持常用名字)
_COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "hot": cv2.COLORMAP_HOT,
    "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
    "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
}

def upsample_and_overlay_heatmap(
    visible_np: np.ndarray,
    heat_lowres: np.ndarray,
    alpha: float = 0.9,
    colormap: str = "rainbow",
    blur_sigma: float = 1.5,
    blur_enable: bool = True,
    interpolation: int = cv2.INTER_CUBIC,
    return_heatmap_bgr: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    将低分辨率热图插值到与 visible_np 相同大小并叠加返回。

    参数:
      visible_np: HWC (H,W,3) 或 CHW (3,H,W) 或 灰度 HxW，dtype 任意（最终转为 uint8）
                  假定 visible 是 OpenCV BGR 格式（如果你使用 RGB，请在外部转回 RGB）。
      heat_lowres: 低分辨率热图，如 (12,12) 的 numpy array（float 或 int）
      alpha: 热图透明强度，0.0-1.0
      colormap: "jet","rainbow","hot","viridis" 等
      blur_sigma: gaussian blur 的 sigma（>0.0 有效）
      blur_enable: 是否在插值后做高斯模糊以去块
      interpolation: cv2.resize 的插值方法（默认为 cubic）
      return_heatmap_bgr: 如果 True，返回 (overlay_bgr, heatmap_bgr)，heatmap_bgr 为插值并上色后的热图

    返回:
      overlay_bgr: uint8 HWC (H, W, 3) BGR
      （可选）heatmap_bgr: uint8 HWC (H, W, 3) BGR（仅 heatmap，未与原图混合）
    """
    
    # --- 规范 visible_np 为 HWC uint8 ---
    vis = visible_np
    if vis.ndim == 3 and vis.shape[0] in (1, 3, 4):  # 可能是 CHW
        if vis.shape[0] in (3, 4):
            vis = np.transpose(vis, (1, 2, 0))
    if vis.ndim == 2:  # 灰度 -> 3通道
        vis = np.stack([vis, vis, vis], axis=-1)
    # 转为 uint8（裁剪）
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)

    H_vis, W_vis = vis.shape[0], vis.shape[1]

    # --- 处理 lowres 热图（单通道） ---
    heat = np.array(heat_lowres, dtype=np.float32)
    print(heat.shape)
    # 如果是多通道 -> collapse 用 max
    if heat.ndim == 3:
        if heat.shape[0] <= 8 and heat.shape[0] != heat.shape[1]:
            heat = np.max(heat, axis=0)
        else:
            heat = np.max(heat, axis=2)

    # 处理 NaN/Inf
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 0..255
    hmin, hmax = float(np.min(heat)), float(np.max(heat))
    
    norm = (heat - hmin) / (hmax - hmin)
    heat_u8 = (norm * 255.0).astype(np.uint8)
    

    # 插值到可见图尺寸（注意 cv2.resize 的 size 参数是 (宽, 高)）
    heat_resized = cv2.resize(heat_u8, (W_vis, H_vis), interpolation=interpolation)
    cv2.imwrite('test1.jpg',heat_resized)

    # 可选模糊以去马赛克（blur_sigma 控制强度）
    if blur_enable and blur_sigma > 0:
        # ksize=(0,0) 让 sigma 决定模糊半径
        heat_resized = cv2.GaussianBlur(heat_resized, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        cv2.imwrite('test2.jpg',heat_resized)

    # 上色：得到 BGR
    cmap_code = _COLORMAPS.get(colormap.lower(), cv2.COLORMAP_RAINBOW)
    heat_color_bgr = cv2.applyColorMap(heat_resized, cmap_code)  # 输出 uint8 BGR
    cv2.imwrite('test3.jpg',heat_color_bgr)

    # alpha 通道基于热强度（0..255），并乘上用户 alpha
    alpha_mask = (heat_resized.astype(np.float32) / 255.0) * float(np.clip(alpha, 0.0, 1.0))
    # 扩展到三通道，用于逐像素加权
    alpha_mask_3ch = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)

    # 转为 float 进行混合
    heat_f = heat_color_bgr.astype(np.float32) / 255.0
    vis_f = vis.astype(np.float32) / 255.0

    out_f = heat_f * alpha_mask_3ch + vis_f * (1.0 - alpha_mask_3ch)
    out_bgr = (np.clip(out_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    if return_heatmap_bgr:
        return out_bgr, heat_color_bgr
    # return out_bgr
    return Image.fromarray(out_bgr, mode="RGB")

def train(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per, tensorboard=None,
          scheduler=None, scaler=None, K_model = None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    criterion = nn.CrossEntropyLoss()
    wq_logit = train_config.infoNCE_logit
    wq_logit = torch.tensor(wq_logit)

    # for loop over one epoch
    for query, reference, ids, labels in bar:

        if scaler:
            with (autocast()):  # -- 使用混合精度
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                # Forward pass
                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    if K_model != None:
                        score_map_q, score_map_r = K_model.forward_feature_map(query), K_model.forward_feature_map(reference)
                    else:
                        score_map_q, score_map_r = None, None
                    # print(score_map_q.shape, score_map_r.shape)
                    output1, output2 = model(query, reference, score_map_q, score_map_r)
                    features1, features2 = output1[-2], output2[-2]  # -- for contrastive
                    features_fine_1, features_fine_2 = output1[-1], output2[-1]  # -- for fine-grained

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                    loss_D_D = loss_functions["infoNCE"](features1, features1, model.module.logit_scale.exp())
                    loss_S_S = loss_functions["infoNCE"](features2, features2, model.module.logit_scale.exp())
                else:
                    # 1. infoNCE
                    loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
                    loss_D_D = loss_functions["infoNCE"](features1, features1, model.logit_scale.exp())
                    loss_S_S = loss_functions["infoNCE"](features2, features2, model.logit_scale.exp())

                    # 2. Fine-grained
                    blocks = train_config.blocks_for_PPB
                    weights = [model.w_blocks1, model.w_blocks2, model.w_blocks3]

                    # ========================================================================================
                    loss_D_fine_S_fine = loss_functions["blocks_mse"](features_fine_1, features_fine_2,
                                                                      model.logit_scale_blocks.exp(), weights,
                                                                      blocks)
                    # ========================================================================================


                    loss_D_fine_D_fine = loss_functions["blocks_infoNCE"](features_fine_1, features_fine_1,
                                                                          model.logit_scale_blocks.exp(), weights,
                                                                          blocks)
                    loss_S_fine_S_fine = loss_functions["blocks_infoNCE"](features_fine_2, features_fine_2,
                                                                          model.logit_scale_blocks.exp(), weights,
                                                                          blocks)

                if train_config.if_learn_ECE_weights:

                    if train_config.if_use_plus_1:
                        if train_config.only_DS:
                            lossall = train_config.weight_D_S * loss + \
                                      model.ECE_weight_D_D * loss_D_D + \
                                      (1 - model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0. * loss_D_fine_D_fine + \
                                      0. * loss_S_fine_S_fine

                        elif train_config.only_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      0. * loss_D_D + \
                                      0. * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      (1 - model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                        elif train_config.DS_and_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      model.ECE_weight_D_D * loss_D_D + \
                                      (1 - model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      (1 - model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                    elif train_config.if_use_multiply_1:
                        if train_config.only_DS:
                            lossall = train_config.weight_D_S * loss + \
                                      0.5 * model.ECE_weight_D_D * loss_D_D + \
                                      0.5 * (1/model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0. * loss_D_fine_D_fine + \
                                      0. * loss_S_fine_S_fine

                        elif train_config.only_fine:
                            # print('use_only_fine')
                            lossall = train_config.weight_D_S * loss + \
                                      0. * loss_D_D + \
                                      0. * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0.5 * model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      0.5 * (1/model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                        elif train_config.DS_and_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      0.5 * model.ECE_weight_D_D * loss_D_D + \
                                      0.5 * (1/model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0.5 * model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      0.5 * (1/model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                else:
                    lossall = train_config.weight_D_S * loss + \
                            train_config.weight_S_S * loss_S_S + \
                            train_config.weight_D_D * loss_D_D + \
                            train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                            train_config.weight_D_fine_D_fine * loss_D_fine_D_fine + \
                            train_config.weight_S_fine_S_fine * loss_S_fine_S_fine


                losses.update(lossall.item())

            # scaler.scale(loss).backward()
            scaler.scale(lossall).backward()
            # print(f"\n=================pos_scale:{model.model_1.pos_scale}====================")

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            # tst = model.logit_scale
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

            if tensorboard is not None:
                steps = step + (epoch - 1) * train_steps_per
                tensorboard.add_scalar("Loss", lossall.item(), steps)
                tensorboard.add_scalar("Loss_Avg", losses.avg, steps)
                tensorboard.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], steps)
                tensorboard.add_scalar("Learning_Rate_Temp", optimizer.param_groups[-1]['lr'], steps)
                tensorboard.add_scalar("Temperature", model.logit_scale.detach().cpu().numpy(), steps)

        step += 1
        # break

    if train_config.verbose:
        bar.close()

    print("/n================================================")
    print("D_D:{}", model.ECE_weight_D_D)
    # print("S_S:{}", model.ECE_weight_S_S)
    print("D_fine_D_fine:{}", model.ECE_weight_D_fine_D_fine)
    # print("S_fine_S_fine:{}", model.ECE_weight_S_fine_S_fine)
    print("================================================")
    return losses.avg


def predict(train_config, model, dataloader, K_model = None, flag = 'query'):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    if flag != 'query':
        if train_config.local_test_save_memory:
            # img_features_part_loacl = []
            pass
        else:
            img_features_list_loacl = []
        
    # os.makedirs(train_config.save_local_feature_path, exist_ok= True)
    ids_list = []
    paths_list = []

    # count = 0
    
    with torch.no_grad():
        # if train_config.local_test_save_memory:
        #     print('bar', len(bar))
        #     partition = 10
        #     part_number = len(bar) // partition
        #     print(part_number)
        #     part_index = 0

        # img_feature_local_total = torch.zeros([len(bar),train_config.batch_size_eval, 1024, 12, 12]).to(train_config.device)
        # img_feature_total = torch.zeros([len(bar),train_config.batch_size_eval, 1024]).to(train_config.device)

        # print(img_feature_local_total.shape, img_feature_total.shape)
        if train_config.plt_sim_heat == True:

            paths_list_CAMP = []
            img_features_list_loacl_CAMP = []
            # need_index = [2207, 5894, 9829]
            # need_index = [266, 2126, 4987]

            # need_index = []
            # test_path  = '/home/lhg/work/ssd_new/visual_geolocalization/LKGL/LKGL_11.24/plot/LKGL_case_study/SUES-200/U1652-D2S/150/query_img.json'
            # with open(test_path, "r") as f:
            #     test_json_list = json.load(f)
            # for pair in test_json_list:
            #     need_index.append(pair['query_index'])

            need_index = []
            test_path  = '/home/lhg/work/ssd_new/visual_geolocalization/LKGL/LKGL_11.24/plot/LKGL_case_study/U1652/U1652-D2S/query_img.json'
            with open(test_path, "r") as f:
                test_json_list = json.load(f)
            for pair in test_json_list:
                need_index.append(pair['query_index'])
            
            # print('need_index', need_index)

            if train_config.dataset_name == 'SUES-200':
                json_path = os.path.join('plot/LKGL_case_study',train_config.dataset_name,train_config.dataset,str(train_config.altitude),'query_img.json')
            else:
                json_path = os.path.join('plot/LKGL_case_study',train_config.dataset_name,train_config.dataset,'query_img.json')
            with open(json_path, "r") as f:
                json_list = json.load(f)

            plt_LKGL_path = []
            plt_CAMP_path = []

            plt_LKGL_g = []
            plt_CAMP_g = []
            for pair in json_list:
                if pair['query_index'] not in need_index:
                    continue
                else:
                   plt_LKGL_path.append(pair["query"]) 
                   plt_LKGL_path.extend(pair["top5_all_paths"])
                   plt_LKGL_g.extend(pair["top5_all_paths"])

                   plt_CAMP_path.append(pair["query"])
                   plt_CAMP_path.extend(pair["top5_global_paths"])
                   plt_CAMP_g.extend(pair["top5_global_paths"])
                
        # print('len plt path', len(plt_LKGL_path), len(plt_CAMP_path))
        index = 0
        for img, ids, paths in bar:

            # plt_path = ['/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/300/query_drone/0059/10.jpg',
            #  '/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/300/gallery_satellite/0059/0.png']

            # plt_path = ['/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/query_drone/0072/42.jpg',
            #         "/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/gallery_satellite/0072/0.png",
            #     "/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/gallery_satellite/0042/0.png",
            #     "/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/gallery_satellite/0130/0.png",
            #     "/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/gallery_satellite/0075/0.png",
            #     "/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512/Testing/150/gallery_satellite/0018/0.png"]
            if train_config.plt_sim_heat == True:
                # print(paths)
                if paths[0] not in plt_LKGL_path:         
                    continue
                else:
                    # print(paths[0])
                    print('draw!!!!!!!!!')
            
            # print(paths)
            # print('img', img.shape)
            # print('ids', ids.shape) # [8]
            # print('ids', ids)
            # count+=1
            
            ids_list.append(ids)
            # if train_config.local_test_save_memory:
            #     if (index + 1) % part_number == 0:
            #         img_features_part_loacl = []

            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    if K_model != None:
                        score_map = K_model.forward_feature_map(img)
                    else:
                        score_map = None
                    
                    img_feature = model(img, score_map)[-2]

                    if train_config.add_local_test:
                                                                                                                                                                
                        img_feature_local = model(img, score_map)[-1]
                    
                    del score_map
                    # img_fine_grained = model(img)[0]
                    # print()

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
                    # torch.save(img_feature, './test_save/1.pth')
                    if train_config.add_local_test:
                        img_feature_local = F.normalize(img_feature_local, dim=1)
                        if train_config.plt_mode != 'none':
                            cmap, dpi, alpha = 'rainbow', 600, 0.5
                            name1, name2, name3 = paths[0].split('/')[-3], paths[0].split('/')[-2], paths[0].split('/')[-1]
                            name_image= name3.split('.')[0]
                            root = './plot/save_heat_maps/'+ train_config.dataset_name +'/'+train_config.dataset+'/'+name1+'/'+name2+'/'
                            os.makedirs(root, exist_ok= True)
                            base = root + name_image
                            if isinstance(img_feature_local, torch.Tensor) and img_feature_local.ndim > 3:
                                draw_feature = img_feature_local[0].cpu()
                                g = tsne_reduce(draw_feature, 12, seed=0)
                            else:
                                draw_feature = img_feature_local.cpu()
                                g = tsne_reduce(draw_feature, 12, seed=0)
                            # print(g.shape)
        
                            if train_config.plt_mode in ("surface", "both"):
                                out_png = f"{base}_surface.png"
                                # print(f"Saving surface image to {out_png} ...")
                                plot_surface_grid_save(g, out_png, cmap=cmap, dpi=dpi)
                            if train_config.plt_mode in ("overlay", "both"):
                                out_overlay = f"{base}_overlay.png"
                                out_heat = f"{base}_heat.png"
                                # print(f"Creating overlay image to {out_overlay} ...")
                                # render surface and heatmap to memory images with same figsize/dpi
                                figsize = (6, 6)
                                surface_img = render_surface_to_image(g, cmap=cmap, dpi=dpi, figsize=figsize)
                                heat_img = render_heatmap_to_image(g, cmap=cmap, alpha=1, dpi=dpi, figsize=figsize)
                                heat_img.save(out_heat, format="PNG") 
                                # cv2.imwrite('test2.jpg', heat_img)
                                img_init = cv2.imread(paths[0])
                                # comp = composite_visible_and_heatmap(img_init, heat_img)
                                comp = upsample_and_overlay_heatmap(img_init, g)
                                
                                comp.save(out_overlay, format="PNG") 
                        # torch.save(img_feature_local, './test_save/2.pth',)
                        
                # print('img_feature', img_feature.shape)
                # print('img_feature_local', img_feature_local.shape)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
                        
            # img_feature_total[index, :, :] = img_feature.to(torch.float32)
            del img_feature

            paths_list.append(paths)
            

            if train_config.add_local_test:
                if flag != 'query':
                    if train_config.local_test_save_memory:
                        torch.save(img_feature_local.to(torch.float32), os.path.join(train_config.save_local_feature_path, 'gallery', str(index)+'.pth'))
                        # img_features_part_loacl.append(img_feature_local.to(torch.float32))
                        # if (index + 1) % part_number == 0:
                        #     img_feature_local = torch.cat(img_features_part_loacl, dim=0)
                        #     torch.save(img_feature_local.to(torch.float32), os.path.join(train_config.save_local_feature_path, 'gallery', str(part_index)+'.pth'))
                        #     img_features_part_loacl = []
                        #     part_index += 1
                    else:
                        img_features_list_loacl.append(img_feature_local.to(torch.float32))
                else:
                    torch.save(img_feature_local.to(torch.float32), os.path.join(train_config.save_local_feature_path, 'query', str(index)+'.pth'))
                
                index += 1

                # img_feature_local_total[index, :, :, :, :] = img_feature_local.to(torch.float32)
                del img_feature_local

            
            # if count >= 5:
            #     break
        if train_config.plt_sim_heat == True:
            for img, ids, paths in bar:
                if paths[0] not in plt_CAMP_path:         
                    continue
                else:
                    print('draw!!!!!!!!!')
                with autocast():
                    img = img.to(train_config.device)

                    if train_config.handcraft_model is not True:
                        img_feature = model(img)
                    else:
                        if K_model != None:
                            score_map = K_model.forward_feature_map(img)
                        else:
                            score_map = None
                        
                        img_feature = model(img, score_map)[-2]

                        if train_config.add_local_test:
                                                                                                                                                                    
                            img_feature_local = model(img, score_map)[-1]
                        
                        del score_map
                        # img_fine_grained = model(img)[0]
                        # print()

                    # normalize is calculated in fp32
                    if train_config.normalize_features:
                        img_feature = F.normalize(img_feature, dim=-1)
                        # torch.save(img_feature, './test_save/1.pth')
                        if train_config.add_local_test:
                            img_feature_local = F.normalize(img_feature_local, dim=1)
                paths_list_CAMP.append(paths)
                img_features_list_loacl_CAMP.append(img_feature_local.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        # img_features = img_feature_total.contiguous().view(len(bar) * train_config.batch_size_eval, 1024
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

        # print('len_____', len(paths_list))
        paths_list = [item for sublist in paths_list for item in sublist]

        if train_config.add_local_test:
            if flag != 'query':
                if train_config.local_test_save_memory:
                    pass
                else:
                    img_features_local = torch.cat(img_features_list_loacl, dim=0)
            # img_features_local = img_feature_local_total.contiguous().view(len(bar) * train_config.batch_size_eval, 1024, 12, 12)

    if train_config.verbose:
        bar.close()

    if train_config.add_local_test:
        if flag != 'query' and train_config.local_test_save_memory == False:
            if train_config.plt_sim_heat == True:
                paths_list_CAMP = [item for sublist in paths_list_CAMP for item in sublist]
                img_features_local_CAMP = torch.cat(img_features_list_loacl_CAMP, dim=0)

                new_local = []
                for  i in  plt_LKGL_g:
                    index_LKGL = paths_list.index(i)
                    new_local.append(img_features_local[index_LKGL])
                new_local = torch.stack(new_local, dim = 0)

                new_local_CAMP = []
                for  i in  plt_CAMP_g:
                    index_CAMP = paths_list_CAMP.index(i)
                    new_local_CAMP.append(img_features_local_CAMP[index_CAMP])
                new_local_CAMP = torch.stack(new_local_CAMP, dim = 0)

                print(len(plt_LKGL_g), len(plt_CAMP_g), new_local.shape, new_local_CAMP.shape)

                print('gallery_path_len', len(paths_list), len(paths_list_CAMP))
                # return img_features, img_features_local, img_features_local_CAMP, ids_list, paths_list, paths_list_CAMP
                return img_features, new_local, new_local_CAMP, ids_list, plt_LKGL_g, plt_CAMP_g
            else:
                return img_features, img_features_local, ids_list, paths_list
        else:
            # print('query_path_len', len(paths_list))
            return img_features, ids_list, paths_list
    else:
        return img_features, ids_list, paths_list
