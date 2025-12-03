# -*- coding: utf-8 -*-
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import cv2
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from segment_anything.modeling.mask_decoder import MLP
from numba.core.typing.builtins import Print
join = os.path.join
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import monai
from torch.cuda.amp import autocast, GradScaler
from matplotlib import font_manager
from segment_anything import sam_model_registry
import argparse
import random
from datetime import datetime
import shutil
import glob
from metrics import *   # 这里应包含 calculate_iou, calculate_dice, sensitivity
from mutiExperts import *
from multi_Modify_structure_ import *
from multi_loss import *

plt.ioff()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# -------------------- 环境线程数 --------------------
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# -------------------- 全局超参 --------------------
input_size = 256  # buildsam imgencoder
accumulation_steps = 4
defer_threshold = 0.5

# %% 参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--tr_npy_path",
    type=str,
    default="../MedSAM/newData/training_slices",
    help="训练npy文件的路径；包含两个子文件夹：gts和imgs",
)
parser.add_argument("--te_npy_path", type=str, default="/data1/seg_code/model_ViT/MedSAM/newData/test_slices",
                    help="测试集 npy 根目录（含 imgs/ 与 gts/）")
parser.add_argument("-task_name", type=str, default="3expert_deferralseg_promise12")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="/data1/seg_code/model_ViT/PROMISE12_l2d_medsam/work_dir/MedSAM/medsam_vit_b.pth")
parser.add_argument("-work_dir", type=str, default="./deferralseg_promise12_numE_sameAC")
# 多次运行（多seed）
parser.add_argument("--seeds", type=str, default="2023,",
                    help="多个随机种子，用英文逗号分隔，如2023, 2025,2027")
parser.add_argument("--aggregate_only", action="store_true",
                    help="仅聚合已有每次运行的 run_summary.json（调试用）")

parser.add_argument("--eval_only", action="store_true",
                    help="仅评估：加载已训练的 best.pth 在测试集上跑")

parser.add_argument("--device", type=str, default="cuda:3")
args = parser.parse_args()

device = torch.device(args.device)

# 合成专家
# experts = [
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.92, bg_acc=0.93, edge_boost=0.08, block_size=2),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.85, bg_acc=0.99, edge_boost=0.1, block_size=4),
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.90, bg_acc=0.95, edge_boost=0.06, block_size=8),
# ]
# experts = [
#     # # E0: none strong
#     # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.90, edge_boost=0.08, block_size=4),
#     # # E1: FG strong
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.90, edge_boost=0.08, block_size=4),
#     # # E2: BG strong
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.99, edge_boost=0.08, block_size=4),
#     # # E3: BD strong
#     SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.90, edge_boost=0.20, block_size=4),
#     # # E4: FG+BG strong
#     # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.99, edge_boost=0.08, block_size=4),
#     # # E5: FG+BD strong
#     # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.90, edge_boost=0.20, block_size=4),
#     # # E6: BG+BD strong
#     # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.88, bg_acc=0.99, edge_boost=0.20, block_size=4),
#     # # E7: FG+BG+BD strong
#     # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.97, bg_acc=0.99, edge_boost=0.20, block_size=4),
# ]
#性能一致
experts = [
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.92, bg_acc=0.98, edge_boost=0.05, block_size=2),
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.91, bg_acc=0.99, edge_boost=0.05, block_size=4),
    SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.93, bg_acc=0.97, edge_boost=0.05, block_size=8),
    # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.90, bg_acc=0.97, edge_boost=0.10, block_size=8),
    # SynthExpertSeg(coverage_ratio=1.0, fg_acc=0.94, bg_acc=0.99, edge_boost=0.06, block_size=8),
]

# -------------------- Dataset --------------------
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, for_eval=False):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "**/*.npy"), recursive=True))
        self.gt_path_files = [f for f in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(f)))]
        self.bbox_shift = 0 if for_eval else bbox_shift   # 评测时强制不抖 bbox
        self.for_eval = for_eval                          # ★ 新增
        self.input_size = input_size
        print(f"图像数量: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_raw = np.load(join(self.img_path, img_name), "r", allow_pickle=True)
        if img_raw.shape != (self.input_size, self.input_size):
            img_resized = cv2.resize(img_raw, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_raw
        img_resized = np.transpose(img_resized, (2, 0, 1))
        assert img_resized.shape == (3, self.input_size, self.input_size)
        assert np.max(img_resized) <= 1.0 and np.min(img_resized) >= 0.0

        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        if gt.shape != (self.input_size, self.input_size):
            gt = cv2.resize(gt, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        assert img_name == os.path.basename(self.gt_path_files[index]), "图像与标签名称不匹配"

        if self.for_eval:
            # 评测：使用全前景（>0），不随机挑类
            gt2D = (gt > 0).astype(np.uint8)
        else:
            # 训练：随机挑一个存在的前景类
            label_ids = np.unique(gt)[1:]
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))

        assert np.max(gt2D) == 1 and np.min(gt2D) == 0

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_resized).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# -------------------- 模型（含 L2D） --------------------
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
        low_res_masks, iou_score, pass_pred, pass_predictor = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks, iou_score, pass_pred, pass_predictor

@torch.no_grad()
def evaluate_on_dataset(medsam_model, dataset, device, save_dir, use_defer=True):
    from math import sqrt
    medsam_model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    per_case = []  # [{name, dice, iou, sens}, ...]
    for image, gt2D, boxes, img_name in tqdm(loader, desc="Evaluating"):
        image = image.to(device, non_blocking=True)
        gt2D  = gt2D.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)

        # 前向
        pred_low, iou_score, pass_softmax, pass_predictor = medsam_model(image, boxes)
        pass_pred = F.interpolate(pass_softmax, size=(pred_low.shape[2], pred_low.shape[3]),
                                  mode='bilinear', align_corners=False)

        if use_defer:
            (final_prediction, expert_pred_section, model_pred_section,
             _, _, _, _, _, _, _,_) = apply_defer_learning_multi(
                experts, pred_low, pass_pred, gt2D, img_name, epoch=-1, model_save_path=save_dir
            )
            pred_bin = (final_prediction > 0.5).float()
        else:
            pred_bin = (torch.sigmoid(pred_low) > 0.5).float()

        d = calculate_dice(pred_bin, gt2D.float()).item()
        j = calculate_iou(pred_bin, gt2D.float()).item()
        s = sensitivity(pred_bin, gt2D.float()).item()
        name = img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name)
        per_case.append({"name": name, "dice": d, "iou": j, "sens": s})

    # 统计量
    def summarize(vals):
        import numpy as np
        n = len(vals)
        mean = float(np.mean(vals)) if n else 0.0
        var  = float(np.var(vals, ddof=1)) if n > 1 else 0.0
        std  = float(np.sqrt(var))
        ci95 = [float(mean - 1.96 * std / sqrt(n)) if n > 1 else mean,
                float(mean + 1.96 * std / sqrt(n)) if n > 1 else mean]
        return {"n": n, "mean": mean, "var": var, "std": std, "ci95": ci95}

    dice_list = [x["dice"] for x in per_case]
    iou_list  = [x["iou"]  for x in per_case]
    sens_list = [x["sens"] for x in per_case]

    agg = {"dice": summarize(dice_list),
           "iou":  summarize(iou_list),
           "sens": summarize(sens_list)}

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_percase_metrics.json"), "w") as f:
        json.dump(per_case, f, indent=2)
    with open(os.path.join(save_dir, "test_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    return agg, per_case

# -------------------- 单次运行（单 seed） --------------------
def run_one(seed: int, base_save_dir: str):
    set_seed(seed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(base_save_dir, f"{args.task_name}-s{seed}-{run_id}")
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.image_encoder.gradient_checkpointing = True

    # 扩展解码器：增加 pass 类别+predictor
    medsam_model.mask_decoder = add_pass_category_and_predictor_to_mask_decoder(
        medsam_model.mask_decoder, num_experts=len(experts), transformer_dim=256
    )

    # 载入预训练/已有权重
    MODEL_PATH = "/data1/seg_code/model_ViT/PROMISE12_l2d_medsam/work_dir/MedSAM-onNewData-20250512-2218/medsam_model_best.pth"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    medsam_model.load_state_dict(checkpoint, strict=False)
    medsam_model.to(device)

    # 冻结 image/prompt encoder
    for p in medsam_model.image_encoder.parameters():
        p.requires_grad = False
    for p in medsam_model.prompt_encoder.parameters():
        p.requires_grad = False

    # 仅训练解码器的部分权重
    for name, param in medsam_model.named_parameters():
        if 'mask_decoder.mask_tokens' in name or 'mask_decoder.output_hypernetworks_mlps' in name:
            param.requires_grad = True

    medsam_model.train()

    print("总参数数量: ", sum(p.numel() for p in medsam_model.parameters()))
    print("可训练参数数量: ", sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, medsam_model.parameters()),
        lr=1e-4, weight_decay=0.01
    )
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

    dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    num_epochs = 1000
    train_dataset = NpyDataset(args.tr_npy_path)
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True,
    )

    patience = 100
    patience_counter = 0
    best_defer_ratio = float('inf')
    defer_patience = 50
    defer_counter = 0
    # 设定负载上限和下限 (K+1 是专家+模型)
    epsilon_upper = torch.tensor([0.35,0.3, 0.31,1.0], device=device)
    epsilon_lower = torch.tensor([0.15, 0.2,0.18,0.0], device=device)

    # 三项最佳指标记录
    best_dice, best_dice_epoch = -float('inf'), -1
    best_iou,  best_iou_epoch  = -float('inf'), -1
    best_sens, best_sens_epoch = -float('inf'), -1

    metrics_file = os.path.join(model_save_path, args.task_name + "_metrics.json")
    scaler = GradScaler()

    for epoch in range(num_epochs):
        epoch_human_loss = 0.0
        epoch_model_predsection_loss = 0.0

        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_sens = 0.0

        expert_iou_sum = 0.0
        model_iou_sum = 0.0
        expert_dice_sum = 0.0
        model_dice_sum = 0.0

        model_dice_full_sum = 0.0
        model_iou_full_sum = 0.0

        defer_area_ratioSum = 0.0

        num_steps = 0

        for step, (image, gt2D, boxes, img_names) in enumerate(tqdm(train_dataloader)):
            num_steps += 1
            image, gt2D = image.to(device, non_blocking=True), gt2D.to(device, non_blocking=True)
            boxes = boxes.to(device)

            with autocast():
                medsam_pred, iou_score, pass_softmax, pass_predictor = medsam_model(image, boxes)

                pass_pred = F.interpolate(pass_softmax, size=(medsam_pred.shape[2], medsam_pred.shape[3]),
                                          mode='bilinear', align_corners=False)

                (final_prediction, expert_pred_section, model_pred_section,
                 expert_section_loss, model_section_loss,
                 expert_section_iou, model_section_iou,
                 expert_section_dice, model_section_dice,
                 expert_cover, epreds) = apply_defer_learning_multi(
                        experts, medsam_pred, pass_pred, gt2D, img_names, epoch, model_save_path
                )

                medsam_pred_sigmoid = torch.sigmoid(medsam_pred)
                model_iou_full = calculate_iou(medsam_pred_sigmoid, gt2D)
                model_dice_full = calculate_dice(medsam_pred_sigmoid, gt2D)

                lambda_defer = (2 + epoch * 0.01)
                penalty, defer_area_ratio = add_defer_area_penalty(
                    expert_pred_section, expert_cover, gt2D, lambda_defer, boxes,
                    min_defer_ratio=0.001, lambda_wrong_defer=20.0
                )

                pseudo_defer_mask = expert_cover.float()
                pass_predictor_upsampled = F.interpolate(
                    pass_predictor, size=pseudo_defer_mask.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                pass_predictor_loss = F.binary_cross_entropy_with_logits(pass_predictor_upsampled, pseudo_defer_mask)
                softmax_defer_prob = pass_pred[:, :-1, :, :].sum(dim=1, keepdim=True)
                consistency_loss = F.mse_loss(pass_predictor_upsampled, softmax_defer_prob.detach())
                defer_reg = pass_predictor_upsampled.mean()
                defer_guidance_loss = pass_predictor_loss + 0.5 * consistency_loss + 0.1 * defer_reg

                w_collab, w_guidance, w_balance = 1.0, 1.0, 5.0
                loss_collab = softmax_loss_segmentation(epreds, medsam_pred, pass_pred, gt2D, len(experts))
                loss_guidance = defer_guidance_loss
                load_penalty, load_dist = add_load_balancing_penalty(pass_pred, epsilon_upper, epsilon_lower)

                loss = w_collab * loss_collab + w_guidance * loss_guidance + w_balance * load_penalty
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 指标累计
            final_prediction_bin = (final_prediction > 0.5).float()
            current_iou = calculate_iou(final_prediction_bin, gt2D.float())
            current_dice = calculate_dice(final_prediction_bin, gt2D.float())
            current_sens = sensitivity(final_prediction_bin, gt2D.float())

            epoch_iou += current_iou.item()
            epoch_dice += current_dice.item()
            epoch_sens += current_sens.item()

            expert_iou_sum += expert_section_iou.item()
            model_iou_sum += model_section_iou.item()
            expert_dice_sum += expert_section_dice.item()
            model_dice_sum += model_section_dice.item()

            model_dice_full_sum += model_dice_full.item()
            model_iou_full_sum += model_iou_full.item()

            defer_area_ratioSum += defer_area_ratio.item()

            # 释放显存
            del medsam_pred, iou_score, pass_pred, medsam_pred_sigmoid, final_prediction_bin,pass_softmax, pass_predictor
            torch.cuda.empty_cache()

        # ---- epoch 汇总（使用真实步数 num_steps）----
        scheduler.step()

        if num_steps == 0:
            print("警告：本 epoch 未迭代样本。")
            break

        epoch_iou /= num_steps
        epoch_dice /= num_steps
        epoch_sens /= num_steps

        expert_iou_sum /= num_steps
        model_iou_sum /= num_steps
        expert_dice_sum /= num_steps
        model_dice_sum /= num_steps

        model_dice_full_sum /= num_steps
        model_iou_full_sum /= num_steps

        defer_area_ratioSum /= num_steps

        metrics = {
            "epoch": epoch,
            "model_section_loss": float(epoch_model_predsection_loss / num_steps) if num_steps > 0 else 0.0,
            "human_loss": float(epoch_human_loss / num_steps) if num_steps > 0 else 0.0,

            "expert_iou": expert_iou_sum,
            "model_iou": model_iou_sum,
            "iou_final": epoch_iou,
            "model_iou_full": model_iou_full_sum,

            "expert_dice": expert_dice_sum,
            "model_dice": model_dice_sum,
            "dice_final": epoch_dice,
            "model_dice_full": model_dice_full_sum,

            "sens_final": epoch_sens,              # ★ 新增

            "defer_area_ratio": defer_area_ratioSum,
        }
        for i, p in enumerate(load_dist[:-1]):
            metrics[f"load_ratio_expert{i}"] = round(p.item(), 4)
        metrics["load_ratio_model"] = round(load_dist[-1].item(), 4)

        # 逐 epoch 记录
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                json.dump([metrics], f, indent=2)
        else:
            with open(metrics_file, 'r+') as f:
                all_metrics = json.load(f)
                all_metrics.append(metrics)
                f.seek(0)
                json.dump(all_metrics, f, indent=2)

        # —— 用 “epoch 平均 Dice” 作为最佳权重保存标准 —— #
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            best_dice_epoch = epoch
            torch.save(medsam_model.state_dict(), join(model_save_path, "medsam_model_best.pth"))
            print(f"[seed={seed}] 保存最佳模型 @ epoch {epoch}: dice_final={best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 分别记录 IoU / Sens 的最佳数值（不触发保存）
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            best_iou_epoch = epoch
        if epoch_sens > best_sens:
            best_sens = epoch_sens
            best_sens_epoch = epoch

        # 推迟比例的早停文件保持
        if defer_area_ratioSum < best_defer_ratio:
            best_defer_ratio = defer_area_ratioSum
            defer_counter = 0
            torch.save(medsam_model.state_dict(), join(model_save_path, "medsam_model_best_defer_ratio.pth"))
            print(f"保存最小推迟比例模型，当前推迟比率: {best_defer_ratio:.4f}")
        else:
            defer_counter += 1

        if patience_counter >= patience:
            print(f"[seed={seed}] 根据 Dice 早停：连续 {patience} 轮无提升。最佳 epoch={best_dice_epoch}，best_dice={best_dice:.4f}")
            break
        if defer_counter >= defer_patience:
            print(f"[seed={seed}] 根据推迟比例早停：连续 {defer_patience} 轮无下降。最小推迟比率: {best_defer_ratio:.4f}")
            break

    torch.cuda.ipc_collect()

    # 本 run 汇总：写 run_summary.json
    run_summary = {
        "seed": int(seed),
        "save_dir": model_save_path,
        "best_dice_final": float(best_dice),
        "best_dice_epoch": int(best_dice_epoch),
        "best_iou_final": float(best_iou),
        "best_iou_epoch": int(best_iou_epoch),
        "best_sens_final": float(best_sens),
        "best_sens_epoch": int(best_sens_epoch),
    }
    test_agg = None
    if args.te_npy_path and os.path.isdir(args.te_npy_path):
        best_path = join(model_save_path, "medsam_model_best.pth")
        if os.path.isfile(best_path):
            medsam_model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
            medsam_model.to(device)
            # 测试集一般不做 bbox 抖动 & 不随机选标签（建议：gt2D = gt>0）
            test_dataset = NpyDataset(args.te_npy_path, bbox_shift=0, for_eval=True)
            test_agg, _ = evaluate_on_dataset(medsam_model, test_dataset, device,
                                              save_dir=model_save_path, use_defer=True)
        else:
            print(f"[seed={seed}] 未找到最佳权重，跳过测试评估：{best_path}")
    if test_agg is not None:
        run_summary["test_summary"] = test_agg  # 含 dice/iou/sens 的 mean/var/std/ci95

    with open(join(model_save_path, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary

# -------------------- 主入口：多 seed & 聚合 --------------------
def summarize(vals):
    import math
    vals = [float(v) for v in vals]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0, "var": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}
    mean = float(np.mean(vals))
    var = float(np.var(vals, ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(var))
    ci95 = [float(mean - 1.96 * std / np.sqrt(n)) if n > 1 else mean,
            float(mean + 1.96 * std / np.sqrt(n)) if n > 1 else mean]
    return {"n": n, "mean": mean, "var": var, "std": std, "ci95": ci95}

def main():
    seeds = [int(s.strip()) for s in args.seeds.split(",") if len(s.strip()) > 0]
    base_save_dir = args.work_dir

    if args.eval_only:
        assert args.te_npy_path and os.path.isdir(args.te_npy_path), "eval_only 需要 --te_npy_path"

        # 遍历已有 run 目录（每个 seed 的训练输出目录）
        run_summaries = []
        for run_dir in sorted(glob.glob(os.path.join(base_save_dir, f"{args.task_name}-s*-*"))):
            best_path = os.path.join(run_dir, "medsam_model_best.pth")
            if not os.path.isfile(best_path):
                continue

            # 1) 构建模型（跟训练时一致的结构改动）
            sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
            medsam_model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder).to(device)
            medsam_model.mask_decoder = add_pass_category_and_predictor_to_mask_decoder(
                medsam_model.mask_decoder, num_experts=len(experts), transformer_dim=256
            )
            # 2) 加载这个run的best权重
            medsam_model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
            medsam_model.to(device)

            # 3) 测试集评估
            test_dataset = NpyDataset(args.te_npy_path, bbox_shift=0, for_eval=True)
            test_agg, _ = evaluate_on_dataset(medsam_model, test_dataset, device, save_dir=run_dir, use_defer=True)

            # 记录一个简要的 summary（供后面聚合）
            run_summaries.append({"save_dir": run_dir, "test_summary": test_agg})

        # 跨 run/seed 聚合测试结果
        test_means = {"dice": [], "iou": [], "sens": []}
        for rs in run_summaries:
            ts = rs["test_summary"]
            test_means["dice"].append(ts["dice"]["mean"])
            test_means["iou"].append(ts["iou"]["mean"])
            test_means["sens"].append(ts["sens"]["mean"])
        agg_test = {k: summarize(v) for k, v in test_means.items()}
        with open(os.path.join(base_save_dir, f"{args.task_name}_aggregate_test_summary.json"), "w") as f:
            json.dump(agg_test, f, indent=2)
        print("[AGG][TEST] Dice:", agg_test["dice"])
        print("[AGG][TEST] IoU :", agg_test["iou"])
        print("[AGG][TEST] Sens:", agg_test["sens"])
        return  # 结束程序

    if args.aggregate_only:
        run_summaries = []
        for p in glob.glob(join(base_save_dir, "**/run_summary.json"), recursive=True):
            with open(p, "r") as f:
                run_summaries.append(json.load(f))
        if len(run_summaries) == 0:
            print("未找到任何 run_summary.json，无法聚合。")
    else:
        run_summaries = []
        for sd in seeds:
            rs = run_one(sd, base_save_dir)
            run_summaries.append(rs)

    dice_list = [rs.get("best_dice_final") for rs in run_summaries if rs.get("best_dice_final") is not None]
    iou_list  = [rs.get("best_iou_final")  for rs in run_summaries if rs.get("best_iou_final")  is not None]
    sens_list = [rs.get("best_sens_final") for rs in run_summaries if rs.get("best_sens_final") is not None]

    agg = {
        "num_runs": len(run_summaries),
        "seeds": seeds,
        "dice": summarize(dice_list),
        "iou":  summarize(iou_list),
        "sens": summarize(sens_list),
        "runs": run_summaries
    }

    agg_json = join(base_save_dir, f"{args.task_name}_aggregate_summary.json")
    with open(agg_json, "w") as f:
        json.dump(agg, f, indent=2)

    agg_txt = join(base_save_dir, f"{args.task_name}_aggregate_summary.txt")
    with open(agg_txt, "w") as f:
        f.write(f"Runs: {agg['num_runs']}\nSeeds: {seeds}\n")
        def p(name, s):
            f.write(f"{name}: mean={s['mean']:.4f}, var={s['var']:.6f}, std={s['std']:.4f}, 95% CI=[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]\n")
        p("Dice (best per run)", agg["dice"])
        p("IoU  (best per run)", agg["iou"])
        p("Sens (best per run)", agg["sens"])

    print("[AGG] Dice:", agg["dice"])
    print("[AGG] IoU :", agg["iou"])
    print("[AGG] Sens:", agg["sens"])
    print(f"[AGG] 写入: {agg_json} / {agg_txt}")
    # 聚合每个 run 的 test_summary
    test_means = {"dice": [], "iou": [], "sens": []}
    for rs in run_summaries:
        ts = rs.get("test_summary")
        if ts:
            test_means["dice"].append(ts["dice"]["mean"])
            test_means["iou" ].append(ts["iou" ]["mean"])
            test_means["sens"].append(ts["sens"]["mean"])

    agg_test = {k: summarize(v) for k, v in test_means.items()}
    with open(join(base_save_dir, f"{args.task_name}_aggregate_test_summary.json"), "w") as f:
        json.dump(agg_test, f, indent=2)
    print("[AGG][TEST] Dice:", agg_test["dice"])
    print("[AGG][TEST] IoU :", agg_test["iou"])
    print("[AGG][TEST] Sens:", agg_test["sens"])

if __name__ == "__main__":
    main()
