import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import monai
import torch.nn as nn
from typing import List, Tuple

from torch.distributed.elastic.multiprocessing.redirects import redirect

from metrics import *
class SynthExpertSeg:
    """
    用于 2D 医学图像分割的“合成专家”。
    Args
    ----
    coverage_ratio : float      # defer 区域里专家标注的比例
    fg_acc         : float      # 前景像素准确率
    bg_acc         : float      # 背景像素准确率
    edge_boost     : float      # 边缘像素在 fg_acc 基础上再 +edge_boost
    block_size     : int        # 连续标注块尺寸 (像素)
    """
    def __init__(
        self,
        coverage_ratio: float = 0.5,
        fg_acc: float = 0.90,
        bg_acc: float = 0.98,
        edge_boost: float = 0.15,
        block_size: int = 64,
    ):
        self.coverage_ratio = coverage_ratio
        self.fg_acc = fg_acc
        self.bg_acc = bg_acc
        self.edge_boost = edge_boost
        self.block_size = block_size

    # ---------- 公开接口 --------------------------------------------------
    @torch.no_grad()
    def predict_mask(
        self,
        gt: torch.Tensor,              # (B,1,H,W)  {0,1}
        defer_mask: torch.Tensor,      # (B,1,H,W)  {0,1}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        expert_pred : (B,1,H,W) float32  0/1
        expert_mask : (B,1,H,W) float32  本专家实际覆盖到的像素=1
        """
        B, _, H, W = gt.shape
        device = gt.device


        # 2) 整幅图生成预测 (非专家区域可后续用模型覆盖) -------------------
        pred_prob = self._region_accuracy_map(gt)    # 每像素正确概率
        rand_mat  = torch.rand_like(pred_prob)
        expert_pred = torch.where(rand_mat < pred_prob, gt.float(), 1-gt.float())

        # 仅在 expert_mask 内保留预测，其余置 0 -> 方便后续融合
        # expert_mask = F.interpolate(expert_mask, size=(expert_pred.shape[2], expert_pred.shape[3]),
        #                                     mode='bilinear', align_corners=False)
        expert_mask=defer_mask


        return expert_pred, expert_mask

    # ---------- 内部方法 ---------------------------------------------------
    def _region_accuracy_map(self, gt: torch.Tensor) -> torch.Tensor:
        """
        为整张 GT 生成一个“每像素被专家预测正确的概率图”。
        """
        # 前景 / 背景
        fg = gt.bool()
        bg = ~fg

        # 膨胀前景 -> 边缘
        kernel = torch.ones((1,1,7,7), device=gt.device)
        dilated = F.conv2d(gt.float(), kernel, padding=3) > 0
        edge = dilated & bg                    # 近前景的背景视为 edge

        prob = torch.zeros_like(gt, dtype=torch.float32)
        prob[fg]   = self.fg_acc
        prob[bg]   = self.bg_acc
        prob[edge] = torch.clamp(prob[edge] + self.edge_boost, 0., 1.)

        return prob


#   多专家融合 + 推迟逻辑
# ----------------------------------------------------------------
def apply_defer_learning_multi(
    experts:          List,          # List[SynthExpertSeg]
    medsam_pred:      torch.Tensor,  # (B,1,H,W) logits
    pass_preds: torch.Tensor,    # (B, num_experts + 1, H, W) softmax values (for defer)
    gt:               torch.Tensor,  # (B,1,H,W) 0/1
    image_names:      List[str],
    epoch:            int,
    model_save_path:  str,

) -> Tuple[torch.Tensor, ...]:
    """
        每个像素只选择 softmax 后最大通道（专家或模型）进行预测，
        返回最终预测和各项评价指标。
        """
    # ---------- 目录 ----------
    vis_dir = os.path.join(model_save_path, "visualizations", f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)

    B, C, H, W = pass_preds.shape
    num_experts = C - 1


    best_idx = pass_preds.argmax(dim=1)
    with torch.no_grad():
        exp_preds, exp_masks = [], []
        for i, exp in enumerate(experts):
            mask_for_exp = (best_idx == i).float().unsqueeze(1)  # (B,1,H,W)
            ep, _ = exp.predict_mask(gt, mask_for_exp)
            exp_preds.append(ep.detach())
            exp_masks.append(mask_for_exp)

        epreds = torch.stack(exp_preds, dim=0)  # (E,B,1,H,W)
        emasks = torch.stack(exp_masks, dim=0)  # (E,B,1,H,W)

    model_sig = torch.sigmoid(medsam_pred)  # (B,1,H,W)


    all_preds = torch.cat([epreds, model_sig.unsqueeze(0)], dim=0)  # (E+1, B,1,H,W)
    #    best_idx 展开为 one-hot: (E+1, B,1,H,W)
    one_hot = F.one_hot(best_idx, num_classes=num_experts + 1) \
        .permute(3, 0, 1, 2).unsqueeze(2).float()
    #
    final_prediction = (all_preds * one_hot).sum(dim=0)  # (B,1,H,W)

    expert_cover = (best_idx < num_experts).float().unsqueeze(1)  # (B,1,H,W)
    expert_pred_sec = expert_cover * final_prediction
    expert_gt_sec = expert_cover * gt
    model_pred_sec = (1 - expert_cover) * final_prediction
    model_gt_sec = (1 - expert_cover) * gt

    expert_sec_iou = calculate_iou(expert_pred_sec, expert_gt_sec)
    model_sec_iou = calculate_iou(model_pred_sec, model_gt_sec)
    expert_sec_dice = calculate_dice(expert_pred_sec, expert_gt_sec)
    model_sec_dice = calculate_dice(model_pred_sec, model_gt_sec)

    # 6.
    seg_loss_fn = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")
    ce_loss_fn = nn.BCELoss(reduction="mean")
    with torch.cuda.amp.autocast(enabled=False):
        expert_sec_loss = seg_loss_fn(expert_pred_sec, expert_gt_sec) + \
                          ce_loss_fn(expert_pred_sec, expert_gt_sec.float())
        model_sec_loss = seg_loss_fn(model_pred_sec, model_gt_sec) + \
                         ce_loss_fn(model_pred_sec, model_gt_sec.float())
    # -
    for i in range(len(image_names)):
        save_visualization(
            model_pred_mask  = model_sig[i],
            gt_mask          = gt[i],
            final_prediction = final_prediction[i],
            expert_mask      = expert_cover[i],
            img_name         = image_names[i],
            output_dir       = vis_dir,
            emasks           = emasks.permute(1, 0, 2, 3, 4)[i], # 第i个批次的专家掩码
            num_experts      = num_experts
        )

    return (final_prediction, expert_pred_sec, model_pred_sec,
            expert_sec_loss,  model_sec_loss,
            expert_sec_iou,   model_sec_iou,
            expert_sec_dice,  model_sec_dice,
             expert_cover,epreds
            )


def save_visualization(
        model_pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        final_prediction: torch.Tensor,
        expert_mask: torch.Tensor,
        img_name: str,
        output_dir: str,
        emasks: torch.Tensor,
        num_experts: int  # Number of experts
):
    """
    生成六幅子图:
      0 GT
      1 defer 掩码 (红=defer, 绿=nodefer)
      2 final 预测 (青色)
      3 MedSAM 预测
      4 expert区域 (每个专家的区域不同颜色)
    """
    # 获取专家掩码
    expert_np = expert_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    gt_np = gt_mask.squeeze(0).cpu().numpy()
    final_np = (final_prediction.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
    med_np = (model_pred_mask.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)

    # 创建保存路径
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3))

    # -- 0 GT --------------------------------------------------------
    axes[0].imshow(gt_np, cmap="gray")
    axes[0].set_title("GT")
    axes[0].axis("off")

    # -- 1 defer掩码 (红=defer, 绿=nodefer) ------------------------------
    rgb_defer = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    rgb_defer[expert_np == 1] = [255, 0, 0]  # 红色表示专家区域
    rgb_defer[expert_np == 0] = [0,0, 0]  # 黑色表示模型区域
    axes[1].imshow(rgb_defer)
    axes[1].set_title("Expert Coverage(red)")
    axes[1].axis("off")

    # -- 2 final prediction -----------------------------------------
    rgb_final = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    rgb_final[final_np == 1] = [0, 255, 255]  # 青色
    axes[2].imshow(rgb_final)
    axes[2].set_title("Final Prediction")
    axes[2].axis("off")

    # -- 3 MedSAM prediction ----------------------------------------
    axes[3].imshow(med_np, cmap="gray")
    axes[3].set_title("MedSAM Prediction")
    axes[3].axis("off")

    # -- 4 expert区域 (每个专家的区域不同颜色) ------------------------
    rgb_experts = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    color_map = [
        [255, 0, 0],  # 红色表示专家 1
        [0, 255, 0],  # 绿色表示专家 2
        [0, 0, 255],  # 蓝色表示专家 3
        [255, 255, 0],  # 黄色表示专家 4
        [0, 255, 255],  # 青色表示专家 5
        [255, 0, 255],  # 品红色表示专家 6
    ]

    # 为每个专家的掩码分配不同的颜色
    for i in range(num_experts):
        expert_cover = emasks[i].squeeze(0).cpu().numpy().astype(np.uint8)  # 获取每个专家负责的区域
        rgb_experts[expert_cover == 1] = color_map[i % len(color_map)]  # 为每个专家分配不同的颜色
    axes[4].imshow(rgb_experts)
    axes[4].set_title(f"Expert Regions ({num_experts} Experts)")
    axes[4].axis("off")
    # === 添加图例 ===
    legend_patches = []
    for i in range(num_experts):
        color_rgb = np.array(color_map[i % len(color_map)]) / 255.0  # 归一化到 [0, 1]
        legend_patches.append(mpatches.Patch(color=color_rgb, label=f"Expert {i + 1}"))
    # 在可视化图下方或右侧添加图例
    axes[4].legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_viz.png"), dpi=200)
    plt.close()

#多专家 defer penalty (连续误差版)
def add_defer_area_penalty(
    expert_prediction:  torch.Tensor,   # (B,1,H,W) 概率
    expert_cover:       torch.Tensor,   # (B,1,H,W) 0/1
    gt_mask:            torch.Tensor,   # (B,1,H,W)
    lambda_defer:       float,
    boxes:              torch.Tensor,   # (B,4)
    min_defer_ratio:    float,
    lambda_wrong_defer: float,
):
    device = gt_mask.device

    # --- defer 面积惩罚 ------------------------------------------------
    box_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    box_area = box_area.to(device)
    defer_ratio = expert_cover.sum((2,3)) / box_area.clamp(min=1)
    if (defer_ratio < min_defer_ratio).all():
        defer_pen = 0.0
    else:
        ideal = 0.03
        defer_pen = lambda_defer * torch.abs(defer_ratio - ideal).mean()


    # --- defer 区域专家误差 (连续差值) ---------------------------------
    err_pixel  = torch.abs(expert_prediction - gt_mask*expert_cover.float())
    wrong_exp  = err_pixel *  expert_cover
    wrong_ratio= wrong_exp.sum() / ( expert_cover).sum().clamp(min=1)
    wrong_pen  = lambda_wrong_defer * wrong_ratio

    total_penalty = defer_pen + wrong_pen

    print(f"[Defer] area ratio: {defer_ratio.mean().item():.4f} |total_penalty： {total_penalty.item():.4f}| defer_penalty: {defer_pen.item():.3f} | "
          f" wrong_defer_penalty: {wrong_pen.item():.3f}")
    return total_penalty, defer_ratio.mean()

# === 专家可靠度追踪器 =========================================
class ExpertReliability:
    """在线估计专家准确率（EMA）"""
    def __init__(self, num_experts: int, alpha: float = 0.1):
        self.alpha = alpha
        self.acc   = torch.zeros(num_experts)      # EMA 的当前值
        self.eps   = 1e-6

    @torch.no_grad()
    def update(self, expert_correct: torch.Tensor):
        """
        expert_correct: (E,B,1,H,W) 的 0/1 张量 —— 本 batch 内专家的像素级正确标志
        """
        device = expert_correct.device  # 获取 expert_correct 的设备
        self.acc = self.acc.to(device)  # 将 self.acc 移动到相同的设备
        # 对像素求平均得到 batch‑level 准确率，再做 EMA
        batch_acc = expert_correct.mean(dim=(1,2,3,4))   # (E,)
        self.acc  = self.alpha * batch_acc + (1-self.alpha) * self.acc

    def weights(self, device):
        """返回归一化后的专家权重 (E,1,1,1,1)"""
        w = self.acc.to(device) + self.eps
        w = w / w.sum()           # 归一化
        return w.view(-1,1,1,1,1) # 便于广播
