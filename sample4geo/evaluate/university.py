import torch
import numpy as np
from tqdm import tqdm
import time
import gc
from ..trainer import predict
import os
from einops import rearrange
import shutil
import cv2

import math
import json

def getLatitudeAndLongitude(imgPath, configDict):
    # 提取路径中的类别文件夹名作为 key
    if isinstance(imgPath, list):
        posInfo = [configDict[p.split("/")[-2]] for p in imgPath]
    else:
        # print(imgPath)
        posInfo = configDict[imgPath.split("/")[-2]]
    return posInfo

def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=bool)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance

def evaluateSingle(distance, K):
    weight = np.ones(K) - np.array(range(0, K, 1)) / K
    m2 = 1 / np.exp(distance * 5e3)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result

def latlog2meter(lata, loga, latb, logb):
    # EARTH_RADIUS = 6378.137 km
    EARTH_RADIUS = 6378.137
    PI = math.pi
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))
    distance = EARTH_RADIUS * dis * 1000
    return distance

def evaluate_SDM(indexOfTopK, queryIndex, K, path_query, path_gallery, configDict):
    query_path = path_query[queryIndex]
    galleryTopKPath = [path_gallery[i] for i in indexOfTopK[:K]]
    
    queryPosInfo = getLatitudeAndLongitude(query_path, configDict)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath, configDict)
    
    distance = euclideanDistance(queryPosInfo, galleryTopKPosInfo)
    P = evaluateSingle(distance, K)
    return P

def evaluate_MA(indexOfTop1, queryIndex, path_query, path_gallery, configDict):
    query_path = path_query[queryIndex]
    galleryTopKPath = path_gallery[indexOfTop1]
    
    queryPosInfo = getLatitudeAndLongitude(query_path, configDict)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath, configDict)
    
    distance_meter = latlog2meter(queryPosInfo[1], queryPosInfo[0], galleryTopKPosInfo[1], galleryTopKPosInfo[0])
    return distance_meter

# def overlay_score_map_on_image(score_map, image, save_path,
#                                alpha=0.6,
#                                colormap='rainbow',
#                                interp=cv2.INTER_CUBIC):
#     """
#     将 12x12（或任意小尺寸）分数图上色并叠加到图片上并保存。
#     参数:
#         score_map: numpy.ndarray 或 torch.Tensor，2D 或 3D（会 squeeze），可能在 GPU 上
#         image: 已由 cv2.imread 读出的图片（H,W,3 或 3,H,W；dtype uint8 或 float）
#         save_path: 保存路径（例如 "out.png"）
#         alpha: 最大叠加透明度，当 score=1 时的 alpha（0..1），默认 0.6
#         colormap: 'rainbow'（默认）或 opencv 的 colormap 常量，如 cv2.COLORMAP_JET
#         interp: cv2.resize 插值方法（默认 INTER_CUBIC）
#     返回:
#         保存成功则返回 True，否则抛出异常
#     """
#     # --- 1. 处理 score_map（支持 torch.Tensor） ---
#     try:
#         import torch
#         is_torch = isinstance(score_map, torch.Tensor)
#     except Exception:
#         torch = None
#         is_torch = False

#     if is_torch:
#         # move to cpu, detach, numpy
#         score_np = score_map.detach().cpu().numpy()
#     else:
#         score_np = np.array(score_map)

#     # squeeze to 2D
#     score_np = np.squeeze(score_np)
#     if score_np.ndim != 2:
#         raise ValueError(f"score_map 应为 2D（或可 squeeze 为 2D），目前 ndim={score_np.ndim}")

#     # --- 2. 处理 image 兼容性 ---
#     img = np.array(image)  # copy-like
#     if img.ndim == 3 and img.shape[0] in (1,3) and img.shape[2] != 3:
#         # 可能为 CHW -> 转为 HWC
#         img = np.transpose(img, (1, 2, 0))
#     if img.ndim != 3 or img.shape[2] != 3:
#         raise ValueError("image 应为 HxWx3 或 3xHxW 格式，且有 3 个通道")

#     # ensure uint8 in 0..255
#     if img.dtype == np.float32 or img.dtype == np.float64:
#         if img.max() <= 1.0:
#             img = (img * 255.0).clip(0,255).astype(np.uint8)
#         else:
#             img = img.clip(0,255).astype(np.uint8)
#     else:
#         img = img.astype(np.uint8)

#     H, W = img.shape[:2]

#     # --- 3. resize score 到图像大小 ---
#     # 将 score 归一化到 0..1（处理常数图）
#     smin, smax = float(np.nanmin(score_np)), float(np.nanmax(score_np))
#     if np.isfinite(smin) and np.isfinite(smax) and (smax - smin) > 1e-8:
#         score_norm = (score_np - smin) / (smax - smin)
#     else:
#         score_norm = np.zeros_like(score_np, dtype=np.float32)

#     # resize to image size
#     score_resized = cv2.resize(score_norm.astype(np.float32), (W, H), interpolation=interp)

#     # --- 4. 转为 8-bit 并上色 (rainbow) ---
#     # 支持字符串 'rainbow' 或直接传入 cv2.COLORMAP_* 常量
#     if isinstance(colormap, str):
#         cmap_name = colormap.lower()
#         cmap_map = {
#             'rainbow': cv2.COLORMAP_RAINBOW,
#             'jet': cv2.COLORMAP_JET,
#             'hot': cv2.COLORMAP_HOT,
#             'viridis': cv2.COLORMAP_VIRIDIS if hasattr(cv2, 'COLORMAP_VIRIDIS') else cv2.COLORMAP_JET,
#         }
#         cmap = cmap_map.get(cmap_name, cv2.COLORMAP_RAINBOW)
#     else:
#         cmap = colormap

#     score_8u = (score_resized * 255.0).clip(0,255).astype(np.uint8)
#     heatmap_bgr = cv2.applyColorMap(score_8u, cmap)  # 返回 BGR

#     # 可选：让热图更亮（非必须），通过提升对比度
#     # heatmap_bgr = cv2.convertScaleAbs(heatmap_bgr, alpha=1.0, beta=0)

#     # --- 5. 基于 score 强度的像素级 alpha 混合（保持原图清晰且热图在高分处更亮） ---
#     # per-pixel alpha = alpha * score_resized
#     per_alpha = (alpha * score_resized).astype(np.float32)  # shape (H,W), in [0,alpha]
#     per_alpha_3 = np.expand_dims(per_alpha, axis=2)  # (H,W,1)

#     # convert to float for blending
#     img_f = img.astype(np.float32)
#     heat_f = heatmap_bgr.astype(np.float32)

#     blended_f = img_f * (1.0 - per_alpha_3) + heat_f * per_alpha_3
#     blended = np.clip(blended_f, 0, 255).astype(np.uint8)

#     # --- 6. 保存文件 ---
#     ok = cv2.imwrite(save_path, blended)
#     if not ok:
#         raise IOError(f"保存失败: {save_path}")
#     return True

def overlay_score_map_on_image(score_map, image, save_path,
                                        alpha=0.6,
                                        colormap='inferno',   # 更醒目的默认色图
                                        interp=cv2.INTER_CUBIC,
                                        gamma=2.0,            # >1 强调高分（默认2）
                                        stretch_percentiles=(2, 98),  # 去除极端值后线性拉伸
                                        use_clahe=False,      # 是否对 8-bit 强度做 CLAHE
                                        clahe_tile=(8,8),
                                        blend_mode='alpha',   # 'alpha' | 'additive'
                                        additive_boost=1.5,   # 当 blend_mode=='additive' 时的放大倍数
                                        highlight_threshold=None,  # 若不为 None，则在高于阈值处画轮廓
                                        contour_color=(0,255,255),
                                        contour_thickness=2):
    """
    更强对比度/高亮的热力叠加函数。
    主要改动点：
      - 对 score 做 percentile 截断 + gamma 幂律（gamma>1 -> 更突出高分）
      - 可选对 8-bit 强度做 CLAHE（提升局部对比）
      - 将热图转换到 HSV 并按 score 放大饱和度
      - 支持两种混合模式：像素级 alpha（默认）或加法增强（更亮）
    """
    # --- 1. 支持 torch.Tensor ---
    try:
        import torch
        is_torch = isinstance(score_map, torch.Tensor)
    except Exception:
        torch = None
        is_torch = False

    if is_torch:
        score_np = score_map.detach().cpu().numpy()
    else:
        score_np = np.array(score_map)
    score_np = np.squeeze(score_np)
    if score_np.ndim != 2:
        raise ValueError(f"score_map must be 2D after squeeze, got ndim={score_np.ndim}")

    # --- 2. image 兼容性 ---
    img = np.array(image)
    if img.ndim == 3 and img.shape[0] in (1,3) and img.shape[2] != 3:
        img = np.transpose(img, (1,2,0))
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must be HxWx3 or 3xHxW with 3 channels")
    # normalize image to uint8 0..255
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0,255).astype(np.uint8)
        else:
            img = img.clip(0,255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    H, W = img.shape[:2]

    # --- 3. 归一化 + percentile stretch + gamma ---
    s = score_np.astype(np.float32)
    # handle nan
    s = np.nan_to_num(s, nan=0.0)
    pmin, pmax = np.percentile(s, stretch_percentiles)
    if pmax - pmin > 1e-8:
        s = (s - pmin) / (pmax - pmin)
        s = np.clip(s, 0.0, 1.0)
    else:
        s = np.zeros_like(s, dtype=np.float32)
    # gamma：gamma>1 更突出高分；gamma<1 会抬低高低差
    if gamma != 1.0:
        s = np.power(s, gamma)

    # resize to image size
    score_resized = cv2.resize(s.astype(np.float32), (W, H), interpolation=interp)
    # final normalized in [0,1]
    score_resized = np.clip(score_resized, 0.0, 1.0)

    # --- 4. to 8-bit and optional CLAHE ---
    score_8u = (score_resized * 255.0).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=clahe_tile)
        score_8u = clahe.apply(score_8u)

    # --- 5. choose colormap (support names) ---
    if isinstance(colormap, str):
        cmap_name = colormap.lower()
        cmap_map = {
            'rainbow': cv2.COLORMAP_RAINBOW,
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS if hasattr(cv2, 'COLORMAP_VIRIDIS') else cv2.COLORMAP_JET,
            'inferno': cv2.COLORMAP_INFERNO if hasattr(cv2, 'COLORMAP_INFERNO') else cv2.COLORMAP_JET,
            'magma': cv2.COLORMAP_MAGMA if hasattr(cv2, 'COLORMAP_MAGMA') else cv2.COLORMAP_JET,
            'plasma': cv2.COLORMAP_PLASMA if hasattr(cv2, 'COLORMAP_PLASMA') else cv2.COLORMAP_JET,
        }
        cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
    else:
        cmap = colormap

    heatmap_bgr = cv2.applyColorMap(score_8u, cmap)  # BGR

    # --- 6. 增强热图饱和度：把 heatmap->HSV，按 score 放大 S/V ---
    hsv = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s_chan, v_chan = cv2.split(hsv)
    # 增强饱和度/亮度（这里用 score_resized 的非线性映射）
    sat_boost = 1.0 + 1.2 * score_resized  # 例如 score=1 -> *2.2
    val_boost = 1.0 + 0.6 * score_resized
    s_chan = np.clip(s_chan * sat_boost, 0, 255)
    v_chan = np.clip(v_chan * val_boost, 0, 255)
    hsv_enh = cv2.merge([h, s_chan, v_chan]).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2BGR).astype(np.float32)

    # --- 7. blending（两种模式） ---
    per_alpha = (alpha * score_resized).astype(np.float32)  # (H,W)
    per_alpha_3 = np.expand_dims(per_alpha, axis=2)

    img_f = img.astype(np.float32)
    if blend_mode == 'alpha':
        blended_f = img_f * (1.0 - per_alpha_3) + heatmap_bgr * per_alpha_3
    elif blend_mode == 'additive':
        # 加法增强：在高分处把热图“加”上去，使高分更亮更鲜艳
        blended_f = img_f + heatmap_bgr * (per_alpha_3 * additive_boost)
    else:
        # fallback to alpha
        blended_f = img_f * (1.0 - per_alpha_3) + heatmap_bgr * per_alpha_3

    blended = np.clip(blended_f, 0, 255).astype(np.uint8)

    # --- 8. 可选：在高于阈值处画轮廓（强调热点边界） ---
    if highlight_threshold is not None:
        # threshold on resized score
        mask = (score_resized >= float(highlight_threshold)).astype(np.uint8) * 255
        # find contours on mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(blended, contours, -1, contour_color, contour_thickness)

    # --- 9. 保存 ---
    ok = cv2.imwrite(save_path, blended)
    if not ok:
        raise IOError(f"save failed: {save_path}")
    return True


def find_bad_res(query_name, gallery_list, tolerance=10):
    if_bad = True

    query_label = query_name.split("/")[-2]
    for i in range(len(gallery_list)):
        res = gallery_list[i].split("/")[-2]
        if res == query_label and i < tolerance:
            if_bad = False
            break

    return if_bad

def find_mediocre_res(query_name, gallery_list, best_threshold=1, worst_threshold=5):
    if_mediocre = False
    query_label = query_name.split("/")[-2]
    for i in range(len(gallery_list)):
        res = gallery_list[i].split("/")[-2]
        if res == query_label and i == best_threshold-1:
            if_mediocre = False
            return if_mediocre
        elif res == query_label and i >= best_threshold-1 and i <= worst_threshold-1:
            if_mediocre = True
            return if_mediocre
        elif i > worst_threshold-1:
            if_mediocre = False
            break

    return if_mediocre

def get_local_score(qf_local, gf_local):
    qf_local = rearrange(qf_local, 'c h w-> c (h w)')
    gf_local = rearrange(gf_local, 'b c h w-> b c (h w)')

    score_local = torch.einsum('bci,cj->bij',gf_local, qf_local)
    score_local = torch.max(score_local, dim = 1)[0]
    score_local = torch.mean(score_local, dim = -1)
    score_local = score_local.squeeze().cpu().numpy()

    return score_local

def eval_query_w_local(qf, ql, gf, gl, qf_local, gf_local=None, itr=0, vote_path=None, index=None):
    # print('qf_local', qf_local.shape)
    # t1 = time.time()
    score_1 = gf @ qf.unsqueeze(-1)
    # t2 = time.time()
    # print('match_time_global', gf.shape, qf.shape, t2 - t1)
    score_1 = score_1.squeeze().cpu().numpy()
    
    qf_local = rearrange(qf_local, 'c h w-> c (h w)')

    if type(gf_local) == str:
        gf_path = gf_local
        score_local_list = []
        for ii in range(len(os.listdir(gf_path))):
            gf_feature_path = os.path.join(gf_path, str(ii)+'.pth')
            gf_feature = torch.load(gf_feature_path)
            gf_feature = rearrange(gf_feature, 'b c h w-> b c (h w)')
            score_local = torch.einsum('bci,cj->bij',gf_feature, qf_local)
            score_local = torch.max(score_local, dim = 1)[0]
            score_local = torch.mean(score_local, dim = -1)
            score_local_list.append(score_local)

        score_local = torch.cat(score_local_list, dim=0)
        score_local = score_local.squeeze().cpu().numpy()

    else:
        # t3 = time.time()
        gf_local = rearrange(gf_local, 'b c h w-> b c (h w)')
        score_local = torch.einsum('bci,cj->bij',gf_local, qf_local)
        score_local = torch.max(score_local, dim = 1)[0]
        score_local = torch.mean(score_local, dim = -1)
        # t4 = time.time()
        # print('match_time_local', gf_local.shape, qf_local.shape, t4 - t3)
        score_local = score_local.squeeze().cpu().numpy()

    score = score_1 + score_local

    if itr !=0:
        itr_dir= os.path.join(vote_path, 'itr_'+str(itr))
        os.makedirs(itr_dir, exist_ok= True)
        np.save(os.path.join(itr_dir, 'score_'+str(itr)+'_'+str(index)+'.npy'), score)

    # score = score_1 + score_4

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp + (index, )

def evaluate(config,
            model,
            query_loader,
            gallery_loader,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True,
            ckpt_path = None,
            K_model = None):
    
    print("Extract Features:")
    if config.add_local_test:
        # config.local_test_save_memory
        query_local_path = os.path.join(config.save_local_feature_path, 'query')
        gallery_local_path = os.path.join(config.save_local_feature_path, 'gallery')
        os.makedirs(query_local_path, exist_ok= True)
        os.makedirs(gallery_local_path, exist_ok= True)

        img_features_query, ids_query, path_query = predict(config, model, query_loader, K_model, flag='query')
        print('config.local_test_save_memory', config.local_test_save_memory)
        if config.local_test_save_memory == False:
            if config.plt_sim_heat == True:
                img_features_gallery, img_features_gallery_local, img_features_gallery_local_CAMP, ids_gallery, path_gallery, path_gallery_CAMP = predict(config, model, gallery_loader, K_model, flag='gallery')
            else:
                img_features_gallery, img_features_gallery_local, ids_gallery, path_gallery = predict(config, model, gallery_loader, K_model, flag='gallery')
        else:
            img_features_gallery, ids_gallery, path_gallery = predict(config, model, gallery_loader, K_model, flag='gallery')

        if config.plt_sim_heat == True:
            # save_path = './plot/save_new_heat_maps/'
            # os.makedirs(save_path, exist_ok= True)
            print('len(path_gallery)', len(path_gallery))
            for i in range(len(path_gallery)):
                use_query_index = i // 5
                use_draw_query = i % 5
                q_base_name = os.path.basename(path_query[use_query_index])
                q_dir_name = os.path.dirname(path_query[use_query_index])
                q_dir_name = q_dir_name.replace('/home/lhg/work/ssd_new/visual_geolocalization', './plot/save_new_heat_maps_LKGL')
                os.makedirs(q_dir_name, exist_ok= True)
                g_base_name = os.path.basename(path_gallery[i])
                g_dir_name = os.path.dirname(path_gallery[i])
                g_dir_name = g_dir_name.replace('/home/lhg/work/ssd_new/visual_geolocalization', './plot/save_new_heat_maps_LKGL')
                os.makedirs(g_dir_name, exist_ok= True)
                save_gallery_path = os.path.join(g_dir_name, g_base_name)
                save_query_path = os.path.join(q_dir_name, q_base_name)

                draw_gallery_feature = img_features_gallery_local[i]
                gallery_img = cv2.imread(path_gallery[i])

                query_img = cv2.imread(path_query[use_query_index])
                # print(os.path.join(query_local_path, str(use_query_index)+'.pth'))
                draw_query_feature = torch.load(os.path.join(query_local_path, str(use_query_index)+'.pth'))
                draw_query_feature = draw_query_feature[0]

                draw_gallery_feature = rearrange(draw_gallery_feature, 'c h w-> c (h w)')
                draw_query_feature = rearrange(draw_query_feature, 'c h w-> c (h w)')

                scores = torch.einsum('ci,cj->ij',draw_query_feature, draw_gallery_feature)

                score_query_vec = scores.max(dim=1).values
                score_gallery_vec = scores.max(dim=0).values

                score_query_map = score_query_vec.reshape(12, 12)
                score_gallery_map = score_gallery_vec.reshape(12, 12)

                if use_draw_query == 0:
                    overlay_score_map_on_image(score_query_map, query_img, save_query_path, alpha=1)
                overlay_score_map_on_image(score_gallery_map, gallery_img, save_gallery_path, alpha=1)
            

            for i in range(len(path_gallery_CAMP)):
                use_query_index = i // 5
                q_base_name = os.path.basename(path_query[use_query_index])
                q_dir_name = os.path.dirname(path_query[use_query_index])
                q_dir_name = q_dir_name.replace('/home/lhg/work/ssd_new/visual_geolocalization', './plot/save_new_heat_maps_CAMP')
                os.makedirs(q_dir_name, exist_ok= True)
                g_base_name = os.path.basename(path_gallery_CAMP[i])
                g_dir_name = os.path.dirname(path_gallery_CAMP[i])
                g_dir_name = g_dir_name.replace('/home/lhg/work/ssd_new/visual_geolocalization', './plot/save_new_heat_maps_CAMP')
                os.makedirs(g_dir_name, exist_ok= True)
                save_gallery_path = os.path.join(g_dir_name, g_base_name)
                save_query_path = os.path.join(q_dir_name, q_base_name)

                draw_gallery_feature = img_features_gallery_local_CAMP[i]
                gallery_img = cv2.imread(path_gallery_CAMP[i])

                query_img = cv2.imread(path_query[use_query_index])
                draw_query_feature = torch.load(os.path.join(query_local_path, str(use_query_index)+'.pth'))
                draw_query_feature = draw_query_feature[0]

                draw_gallery_feature = rearrange(draw_gallery_feature, 'c h w-> c (h w)')
                draw_query_feature = rearrange(draw_query_feature, 'c h w-> c (h w)')

                scores = torch.einsum('ci,cj->ij',draw_query_feature, draw_gallery_feature)

                # score_query_vec = scores.min(dim=1).values
                # score_gallery_vec = scores.min(dim=0).values
                score_query_vec = scores.mean(dim=1)
                score_gallery_vec = scores.mean(dim=0)

                score_query_map = score_query_vec.reshape(12, 12)
                score_gallery_map = score_gallery_vec.reshape(12, 12)

                overlay_score_map_on_image(score_query_map, query_img, save_query_path, alpha=1)
                overlay_score_map_on_image(score_gallery_map, gallery_img, save_gallery_path, alpha=1)
            
    else:
        img_features_query, ids_query, path_query = predict(config, model, query_loader, K_model)
        img_features_gallery, ids_gallery, path_gallery = predict(config, model, gallery_loader, K_model)
    
    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    if config.itr != 0:
        gl_path, ql_path= os.path.join(config.vote_root_path, 'gl.npy'), os.path.join(config.vote_root_path, 'ql.npy')
        if not os.path.isfile(gl_path):
            np.save(gl_path, gl)
        if not os.path.isfile(ql_path):
            np.save(ql_path, ql)
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    indexOfTopK_list = []
    with open("./mediocre_results_xpw.txt", 'w') as f:

        for i in tqdm(range(len(ids_query))):

            if config.add_local_test:
                index_number = i // config.batch_size_eval
                img_number = i % config.batch_size_eval
                img_batch = torch.load(os.path.join(query_local_path, str(index_number)+'.pth'))
                img_features_query_local = img_batch[img_number, ...]

                if config.local_test_save_memory == False:
                    ap_tmp, CMC_tmp, index_rank = eval_query_w_local(img_features_query[i], ql[i], 
                                                        img_features_gallery, gl,
                                                        img_features_query_local, img_features_gallery_local,
                                                        config.itr, config.vote_root_path, i)
                else:
                    ap_tmp, CMC_tmp, index_rank = eval_query_w_local(img_features_query[i], ql[i], 
                                                        img_features_gallery, gl,
                                                        img_features_query_local, gallery_local_path,
                                                        config.itr, config.vote_root_path, i)
            else:
                ap_tmp, CMC_tmp, index_rank = eval_query(img_features_query[i], ql[i], img_features_gallery, gl, config.vote_root_path, i)

            # -- write list
            # query_name = path_query[i]
            # ref_names = []
            # for j in range(10):
            #     ref_names.append(path_gallery[index_rank[j]])
            # # if find_bad_res(query_name, ref_names, tolerance=10):
            # #     f.write(f"{query_name} {ref_names}\n")

            # if find_mediocre_res(query_name, ref_names, best_threshold=1, worst_threshold=5):
            #     f.write(f"{query_name} {ref_names}\n")
            indexOfTopK_list.append(index_rank)

            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))     

    if config.dataset_name == 'DenseUAV':
        gps_file_path = os.path.join(config.data_folder, "Dense_GPS_ALL.txt")
        
        if os.path.isfile(gps_file_path):
            print("Loading GPS config for SDM/MA computation...")
            configDict = {}
            with open(gps_file_path, "r") as F:
                context = F.readlines()
                for line in context:
                    splitLineList = line.strip().split(" ")
                    configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
                                                                float(splitLineList[2].split("N")[-1])]
            
            print("Computing SDM@K...")
            SDM_dict = {}
            for K_val in tqdm(range(1, 101, 1)):
                metric = 0
                for i in range(len(ids_query)):
                    P_ = evaluate_SDM(indexOfTopK_list[i], i, K_val, path_query, path_gallery, configDict)
                    metric += P_
                metric = metric / len(ids_query)
                
                if K_val in ranks:
                    string.append("SDM@{} = {:.2f}%".format(K_val, metric * 100))
                    print("SDM@{} = {:.2f}%".format(K_val, metric * 100))
                SDM_dict[K_val] = metric

            # print("Computing MA@K...")
            # MA_dict = {}
            # for meter in tqdm(range(1, 101, 1)):
            #     MA_K = 0
            #     for i in range(len(ids_query)):
            #         # index_rank[0] 也就是 indexOfTopK_list[i][0] 是 Top1 的预测
            #         MA_meter = evaluate_MA(indexOfTopK_list[i][0], i, path_query, path_gallery, configDict)
            #         if MA_meter < meter:
            #             MA_K += 1
            #     MA_K = MA_K / len(ids_query)
            #     MA_dict[meter] = MA_K

            # 保存为 json，路径统一放在 config.vote_root_path 下
            # save_sdm_path = os.path.join(config.vote_root_path, "SDM@K(1,100).json")
            # save_ma_path = os.path.join(config.vote_root_path, "MA@K(1,100).json")
            
            # with open(save_sdm_path, 'w') as F:
            #     json.dump(SDM_dict, F, indent=4)
            # with open(save_ma_path, 'w') as F:
            #     json.dump(MA_dict, F, indent=4)
        else:
            print(f"Warning: Could not find GPS config at {gps_file_path}. Skipping SDM and MA evaluation.")        
        
    print(' - '.join(string)) 
    if config.itr != 0:
        with open (os.path.join(config.vote_root_path, 'vote_results.txt'), 'a') as f:
            f.write('*'*8 + str(config.itr) + '*'*8 + '\n')
            f.write(config.save_ckpt_path + '\n')
            f.write('_'.join(string))
            f.write('\n')
            f.close()
    else:
        if config.multi_weather == True and config.only_test == True:
            with open (os.path.join('weather_test_result'+'.txt'), 'a') as f:
                f.write('*'*8 + str(config.weather_condition) + '*'*8 + '\n')
                if config.dataset_name == 'SUES-200':
                    write_ = os.path.join(config.vote_root_path, config.dataset_name, config.dataset, str(config.altitude))
                else:
                    write_ = os.path.join(config.vote_root_path, config.dataset_name, config.dataset)
                f.write(write_ + '\n')
                f.write('_'.join(string))
                f.write('\n')
                f.close()

        else:
            with open (os.path.join('test_once_result.txt'), 'a') as f:
                f.write('*'*8 + str(config.itr) + '*'*8 + '\n')
                if config.dataset_name == 'SUES-200':
                    write_ = os.path.join(config.vote_root_path, config.dataset_name, config.dataset, str(config.altitude))
                else:
                    write_ = os.path.join(config.vote_root_path, config.dataset_name, config.dataset)
                f.write(write_ + '\n')
                f.write('_'.join(string))
                f.write('\n')
                f.close()
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        shutil.rmtree(config.save_local_feature_path)
        try:
            del img_features_query_local
            del img_features_gallery_local
        except:
            pass
        
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf, ql, gf, gl, vote_path=None, index=None):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    np.save(os.path.join(vote_path, 'score_'+str(index)+'.npy'), score)

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp + (index, )


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc




