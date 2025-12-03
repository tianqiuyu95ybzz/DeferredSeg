import torch.nn as nn
import torch
from typing import Tuple
from metrics import *
from segment_anything.modeling.mask_decoder import MLP
from numba.core.typing.builtins import Print
#增加pass类别到mask_decoder的输出层
def add_pass_category_to_mask_decoder(mask_decoder: nn.Module, num_experts: int = 1):
    """
    修改 MaskDecoder，增加多个推迟学习通道（每个专家一个通道，再加一个模型决策通道）
    """
    device = mask_decoder.mask_tokens.weight.device

    # --- 1. 修改 Mask Tokens ---
    original_num_mask_tokens = mask_decoder.num_mask_tokens
    mask_decoder.num_mask_tokens += num_experts + 1  # # 为每个专家和模型决策增加一个通道

    new_mask_tokens = nn.Embedding(mask_decoder.num_mask_tokens, mask_decoder.transformer_dim).to(device)
    new_mask_tokens.weight.data[:original_num_mask_tokens] = mask_decoder.mask_tokens.weight.data.to(device)

    mask_decoder.mask_tokens = new_mask_tokens

    # --- 2. 修改 Hypernetwork MLPs ---
    new_hypernetworks_mlps = nn.ModuleList(
        [
            MLP(mask_decoder.transformer_dim, mask_decoder.transformer_dim, mask_decoder.transformer_dim // 8, 3).to(device)
            for _ in range(mask_decoder.num_mask_tokens)
        ]
    )
    mask_decoder.output_hypernetworks_mlps = new_hypernetworks_mlps

    # --- 3. 修改 forward 方法，使其返回 pass_pred ---
    original_forward = mask_decoder.forward

    def new_forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        pass_preds = masks[:, -num_experts-1:, :, :]  # 获取最后 `num_experts + 1` 个通道作为推迟预测
        pass_preds = torch.softmax(pass_preds, dim=1)
        # print(pass_preds.shape)[2, 4, 256, 256]


        # 强制只取前景掩码 + 推迟掩码
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # pass_pred=torch.sigmoid(pass_pred)
        # print(masks.shape)[2, 1, 256, 256]
        return masks, iou_pred,  pass_preds #新增加一个return

    # 绑定新 forward 方法
    mask_decoder.forward = new_forward.__get__(mask_decoder, type(mask_decoder))

    return mask_decoder



def add_pass_category_and_predictor_to_mask_decoder(mask_decoder: nn.Module, num_experts: int = 1, transformer_dim: int = 256):
    """
    修改 MaskDecoder：
    - 增加 num_experts + 1 个 mask token（用于 softmax defer）
    - 增加一个 PassPredictor 分支（用于显式 pass 概率输出）
    - 修改 forward：返回 masks, iou_pred, pass_preds_softmax, pass_probs_predictor
    """
    device = mask_decoder.mask_tokens.weight.device

    # === 1.  mask tokens ===
    original_num_mask_tokens = mask_decoder.num_mask_tokens
    mask_decoder.num_mask_tokens += num_experts + 1

    new_mask_tokens = nn.Embedding(mask_decoder.num_mask_tokens, mask_decoder.transformer_dim).to(device)
    new_mask_tokens.weight.data[:original_num_mask_tokens] = mask_decoder.mask_tokens.weight.data.to(device)
    mask_decoder.mask_tokens = new_mask_tokens

    # === 2. HyperNetwork MLPs ===
    mask_decoder.output_hypernetworks_mlps = nn.ModuleList([
        MLP(mask_decoder.transformer_dim, mask_decoder.transformer_dim, mask_decoder.transformer_dim // 8, 3).to(device)
        for _ in range(mask_decoder.num_mask_tokens)
    ])

    # === 3.  PassPredictor  ===
    class PassPredictor(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.pass_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, 1, kernel_size=1)
            )
        def forward(self, features):
            return self.pass_head(features)

    mask_decoder.pass_predictor = PassPredictor(transformer_dim).to(device)


    original_predict_masks = mask_decoder.predict_masks

    def new_forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        masks, iou_pred = original_predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # === softmax-based defer ===
        pass_preds_softmax = masks[:, -num_experts - 1:, :, :]  # (B, E+1, H, W)
        pass_preds_softmax = torch.softmax(pass_preds_softmax, dim=1)

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        pass_logits = mask_decoder.pass_predictor(image_embeddings)  # (B,1,H,W)
        pass_probs_predictor = torch.sigmoid(pass_logits)

        return masks, iou_pred, pass_preds_softmax, pass_probs_predictor


    mask_decoder.forward = new_forward.__get__(mask_decoder, type(mask_decoder))

    return mask_decoder
