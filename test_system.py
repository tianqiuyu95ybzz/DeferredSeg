# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import os
import cv2
from debugpy.launcher import channel
join = os.path.join
from segment_anything import sam_model_registry
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import monai
import torch.nn.functional as F
import glob
from torch.utils.data import Dataset, DataLoader
from metrics import *
# from expert import *
# from Modify_structure  import *
from mutiExperts import *
from multi_Modify_structure_ import *
from multi_loss import *
from muti_expertMYtrain_l2d_Single_c_prediction import NpyDataset,MedSAM
# 加载保存的最佳模型参数
MODEL_PATH = "/data1/seg_code/model_ViT/PROMISE12_l2d_medsam/deferralseg_promise12_numE_complementation/5expert_deferralseg_promise12-s2027-20251016-0959/medsam_model_best.pth"
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
experts = [
    # # E0: none strong
    # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.90, edge_boost=0.08, block_size=4),
    # # E1: FG strong
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.90, edge_boost=0.08, block_size=4),
    # # E2: BG strong
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.99, edge_boost=0.08, block_size=4),
    # # E3: BD strong
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.90, edge_boost=0.20, block_size=4),
    # # E4: FG+BG strong
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.99, edge_boost=0.08, block_size=4),
    # # E5: FG+BD strong
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.90, edge_boost=0.20, block_size=4),
    # # E6: BG+BD strong
    # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.99, edge_boost=0.20, block_size=4),
    # # E7: FG+BG+BD strong
    # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.99, edge_boost=0.20, block_size=4),
]
# experts = [
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.92, bg_acc=0.98, edge_boost=0.05, block_size=2),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.91, bg_acc=0.99, edge_boost=0.05, block_size=4),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.93, bg_acc=0.97, edge_boost=0.05, block_size=8),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.90, bg_acc=0.97, edge_boost=0.10, block_size=8),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.94, bg_acc=0.99, edge_boost=0.06, block_size=8),
# ]
def calculate_metrics(pred_mask, gt_mask):
    """
    返回 (IoU, Dice, Sensitivity,
          CRA_IoU, CRA_Dice,
          HD95)
    """
    # ------- 基本指标 --------------------------------------------------
    pred_flat = pred_mask.cpu().numpy().flatten()
    gt_flat   = gt_mask.cpu().numpy().flatten()

    iou  = jaccard_score(gt_flat, pred_flat, average='binary')
    dice = f1_score(gt_flat, pred_flat, average='binary')

    tp  = (pred_flat & gt_flat).sum()
    fn  = gt_flat.sum() - tp
    sens = tp / (tp + fn) if (tp + fn) else 1.0


    return iou, dice, sens


def load_model():
    """加载预训练模型并加载最佳参数."""
    # 使用注册表初始化SAM模型
    sam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
    # 动态修改 MaskDecoder 添加 pass 类别
    model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(DEVICE)
    # 动态修改 MaskDecoder 添加 pass 类别
    # model.mask_decoder = add_pass_category_to_mask_decoder(model.mask_decoder,num_experts=len(experts))
    model.mask_decoder = add_pass_category_and_predictor_to_mask_decoder(
        model.mask_decoder,
        num_experts=len(experts),
        transformer_dim=256
    )
    # 加载权重（仅加载模型的状态字典部分）
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print(model.__class__.__name__)

    model.eval()  # 设置为评估模式
    return model



def test_model(test_loader, model, model_save_path):
    """测试模型并分别输出 system / expert / model 的平均 6 指标"""
    # ------- 累加器  ---------------------------------------------------
    system  = np.zeros(3);  expert = np.zeros(3);  model_m = np.zeros(3)
    num_samples = 0
    epoch = 0

    with torch.no_grad():
        for images, gt_masks, boxes, img_names in tqdm(test_loader):
            images, gt_masks = images.to(DEVICE), gt_masks.to(DEVICE)
            medsam_pred, _, pass_softmax, _ = model(images, boxes)

            pass_pred = F.interpolate(pass_softmax,
                                      size=medsam_pred.shape[-2:],
                                      mode='bilinear', align_corners=False)

            final_prediction, _, _, _, _, _, _, _, _, expert_cover, _ = \
                apply_defer_learning_multi(experts,
                                           medsam_pred, pass_pred,
                                           gt_masks, img_names, epoch,
                                           model_save_path)

            # --------- 逐样本累加 --------------------------------------
            for i in range(final_prediction.size(0)):
                # 1) System 全图
                sys_metrics = calculate_metrics((final_prediction[i] > 0.5).long(),
                                                gt_masks[i])
                system += np.array(sys_metrics)

                # 2) Expert‑only 区域
                expert_mask = (expert_cover[i] == 1).long()
                exp_metrics = calculate_metrics(((final_prediction[i] > 0.5).long() * expert_mask),
                                                gt_masks[i] * expert_mask)
                expert += np.array(exp_metrics)

                # 3) Model‑only 区域
                model_mask = (expert_cover[i] == 0).long()
                mod_metrics = calculate_metrics(((final_prediction[i] > 0.5).long() * model_mask),
                                                gt_masks[i] * model_mask)
                model_m += np.array(mod_metrics)

                num_samples += 1

    # -------- 平均值 ---------------------------------------------------
    avg_sys = (system / num_samples).tolist()
    avg_exp = (expert / num_samples).tolist()
    avg_mod = (model_m / num_samples).tolist()

    names = ["IoU", "Dice", "Sens"]
    def _fmt(vals): return " | ".join(f"{n}:{v:.4f}" if n!="HD95" else f"{n}:{v:.2f}"
                                      for n, v in zip(names, vals))

    print(f"[System] {_fmt(avg_sys)}")
    print(f"[Expert] {_fmt(avg_exp)}")
    print(f"[Model ] {_fmt(avg_mod)}")

    save_results(avg_sys, avg_exp, avg_mod, model_save_path)
    return avg_sys, avg_exp, avg_mod

def save_results(sys_m, exp_m, mod_m, path):
    names = ["IoU", "Dice", "Sens"]
    def _line(title, vals):
        metrics = ", ".join(f"{n}:{vals[i]:.4f}" if n!="HD95" else f"{n}:{vals[i]:.2f}"
                            for i, n in enumerate(names))
        return f"[{title}] {metrics}\n"

    txt = _line("System", sys_m) + _line("Expert", exp_m) + _line("Model ", mod_m)
    with open(os.path.join(path, "test_system_results_num of exp_0.94.txt"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    dataset = NpyDataset("../MedSAM/newData/test_slices")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 加载模型
    model = load_model()
    # 创建保存路径（带时间戳）
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_dir = os.path.dirname(MODEL_PATH)
    model_save_path = os.path.join(model_dir, f"test_{run_id}")
    os.makedirs(model_save_path, exist_ok=True)

    # 测试模型
    test_model(loader, model, model_save_path)
