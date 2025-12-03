import torch
import torch.nn.functional as F
from numpy.ma.core import shape


def softmax_loss_segmentation(epreds, model_pred, pass_preds, labels, num_experts,dice_weight =1.0,eps=1e-8):
    """
    适用于分割任务的 softmax 损失函数，考虑推迟决策。
    ----
    epreds: (n_experts, B, 1, H, W)  # 专家的预测，n_experts个专家的预测
    model_pred: (B, 1, H, W)  # 模型的预测 logits
    pass_preds: (B, n_experts + 1, H, W)  # 推迟给模型或专家的概率（softmax 后的输出）
    labels: (B, 1, H, W)  # 真实标签（0 或 1，二值分割任务）
    num_experts: 专家数量
    eps: 防止除零错误的小常数
    """
    B, _, H, W = pass_preds.shape

    # === 1.
    expert_correct = (epreds.squeeze(2) == labels.squeeze(1).unsqueeze(0)).float()  # (n_experts, B, H, W)
    expert_correct = expert_correct.permute(1, 0, 2, 3)  # (B, n_experts, H, W)
    model_pred = torch.sigmoid(model_pred)  # (B,1,H,W)
    correct_model = (model_pred.squeeze(1) > 0.5) == labels.squeeze(1)  # (B, H, W)

    # === 2
    log_probs = torch.log(pass_preds + eps)  # (B, n_experts+1, H, W)

    # === 3.
    expert_log_probs = log_probs[:, :num_experts, :, :]  # (B, n_experts, H, W)
    loss_expert = torch.sum((1 - 2 * expert_correct) * expert_log_probs, dim=1)  # (B, H, W)

    # === 4.
    model_log_probs = log_probs[:, -1, :, :]  # (B, H, W)
    loss_bce = F.binary_cross_entropy_with_logits(model_pred, labels.float(), reduction='none').squeeze(1)  # (B, H, W)
    loss_model = loss_bce - correct_model.float() * model_log_probs

    # loss_dice = dice_weight * dice_loss(model_pred, labels)  # scalar

    # === 5.
    total_loss = loss_expert.mean()+loss_model.mean() #+ loss_dice
    return total_loss



def dice_loss(logits, targets, eps=1e-8):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def add_load_balancing_penalty(pass_preds: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor, λ: float = 1.0):
    """
    pass_preds: (B, K+1, H, W)
    upper/lower: (K+1,)
    """
    avg_ratio = pass_preds.mean(dim=(0, 2, 3))  # 每个通道平均值
    upper_penalty = F.relu(avg_ratio - upper).sum()
    lower_penalty = F.relu(lower - avg_ratio).sum()
    penalty = λ * (upper_penalty + lower_penalty)
    return penalty, avg_ratio




