# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#修改了两处
import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
          使用SAM计算图像嵌入，并在给定提示的情况下允许重复的高效掩码预测。
        参数:
          sam_model (Sam): 用于掩码预测的模型。
        """
        super().__init__()
        self.model = sam_model  # 将传入的模型赋值给实例变量
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)  # 初始化图像变换工具
        self.reset_image()  # 重置当前图像状态

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
         计算提供图像的图像嵌入，以便使用'predict'方法预测掩码。

        参数:
          image (np.ndarray): 用于计算掩码的图像。期望为HWC格式的uint8图像，像素值范围在[0, 255]。
          image_format (str): 图像的颜色格式，取值为['RGB', 'BGR']。
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."  # 确保图像格式正确
        if image_format != self.model.image_format:  # 如果图像格式与模型不匹配
            image = image[..., ::-1]  # 转换图像格式

        # 将图像转换为模型期望的格式
        input_image = self.transform.apply_image(image)  # 应用图像变换
        input_image_torch = torch.as_tensor(input_image, device=self.device)  # 转换为torch张量
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
                            None, :, :, :
                            ]  # 调整张量维度为1x3xHxW格式

        self.set_torch_image(input_image_torch, image.shape[:2])  # 设置处理后的图像

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
         计算提供图像的图像嵌入，以便使用'predict'方法预测掩码。期望输入图像已经转换为模型期望的格式。

        参数:
          transformed_image (torch.Tensor): 输入图像，形状为1x3xHxW，已经经过ResizeLongestSide处理。
          original_image_size (tuple(int, int)): 图像转换前的大小，以(H, W)格式表示。
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] in[3,4] #三通道或者四通道都可以，原本是=3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()# 重置图像状态

        self.original_size = original_image_size  # 保存原始图像尺寸
        self.input_size = tuple(transformed_image.shape[-2:])  # 保存输入图像尺寸
        self.input_image = self.model.preprocess(transformed_image)  # 预处理图像
        self.features = self.model.image_encoder(self.input_image)  # 提取图像特征
        self.is_image_set = True  # 设置图像已设置标志

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
         根据当前设置的图像预测掩码。

        参数:
          point_coords (np.ndarray or None): Nx2的点提示数组。每个点以(X,Y)格式表示。
          point_labels (np.ndarray or None): 点提示的标签数组，长度为N。1表示前景点，0表示背景点。
          box (np.ndarray or None): 长度为4的数组，给出模型的框提示，格式为XYXY。
          mask_input (np.ndarray): 模型的低分辨率掩码输入，通常来自于前一次预测。形状为1xHxW，SAM中H=W=256。
          multimask_output (bool): 如果为真，模型将返回三个掩码。
          return_logits (bool): 如果为真，返回未经阈值处理的掩码logits而不是二进制掩码。

        返回:
          (np.ndarray): 形状为CxHxW的输出掩码，其中C是掩码的数量，(H, W)是原始图像大小。
          (np.ndarray): 长度为C的数组，包含模型对每个掩码质量的预测。
          (np.ndarray): 形状为CxHxW的数组，其中C是掩码的数量，H=W=256。这些低分辨率logits可以传递给后续迭代作为掩码输入。
        """
        if not self.is_image_set: # 检查图像是否已设置
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # 变换输入提示
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:  # 如果提供了点坐标
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."  # 标签必须存在
            point_coords = self.transform.apply_coords(point_coords, self.original_size)  # 应用坐标变换
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device  # 转换为torch张量
            )
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device
            )  # 转换标签为torch张量
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 添加批次维度
        if box is not None:  # 如果提供了框提示
            box = self.transform.apply_boxes(box, self.original_size)  # 应用框变换
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)  # 转换为torch张量
            box_torch = box_torch[None, :]  # 添加批次维度
        if mask_input is not None:  # 如果提供了低分辨率掩码输入
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device
            )  # 转换为torch张量
            mask_input_torch = mask_input_torch[None, :, :, :]  # 添加批次维度

        # 进行掩码预测
        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # masks_np = masks[0].detach().cpu().numpy()
        # iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        # return masks_np, iou_predictions_np, low_res_masks_np
        # 返回包含批次维度的完整输出，可以直接返回整个张量而不提取第一个批次：
        masks_np = masks.detach().cpu().numpy()  # 将掩码转换为NumPy数组
        iou_predictions_np = iou_predictions.detach().cpu().numpy()  # 将IoU预测转换为NumPy数组
        low_res_masks_np = low_res_masks.detach().cpu().numpy()  # 将低分辨率掩码转换为NumPy数组
        return masks_np, iou_predictions_np, low_res_masks_np  # 返回掩码和预测结果

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         根据当前设置的图像和输入提示预测掩码。
        输入提示为批处理的torch张量，期望已使用ResizeLongestSide转换为输入格式。

        参数:
          point_coords (torch.Tensor or None): BxNx2的点提示数组。每个点以(X,Y)格式表示。
          point_labels (torch.Tensor or None): BxN数组，表示点提示的标签。1表示前景点，0表示背景点。
          boxes (np.ndarray or None): Bx4数组，给出模型的框提示，格式为XYXY。
          mask_input (np.ndarray): 低分辨率掩码输入，形状为Bx1xHxW，通常为256x256。
          multimask_output (bool): 如果为真，模型将返回三个掩码。
          return_logits (bool): 如果为真，返回未经阈值处理的掩码logits而不是二进制掩码。

        返回:
          (torch.Tensor): 形状为BxCxHxW的输出掩码，其中C是掩码的数量，(H, W)是原始图像大小。
          (torch.Tensor): 形状为BxC的数组，包含模型对每个掩码质量的预测。
          (torch.Tensor): 形状为BxCxHxW的数组，其中C是掩码的数量，H=W=256。这些低分辨率logits可以传递给后续迭代作为掩码输入。
        """
        if not self.is_image_set:  # 检查图像是否已设置
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:  # 如果提供了点坐标
            points = (point_coords, point_labels)  # 保存点坐标和标签
        else:
            points = None  # 点提示为None

            # 嵌入提示
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )  # 使用模型的提示编码器嵌入提示

        # 预测掩码
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )  # 使用掩码解码器预测掩码

        # 将掩码上采样到原始图像分辨率
        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )  # 处理低分辨率掩码

        if not return_logits:  # 如果不返回logits
            masks = masks > self.model.mask_threshold  # 应用阈值获取二进制掩码

        return masks, iou_predictions, low_res_masks  # 返回掩码和预测结果

    def get_image_embedding(self) -> torch.Tensor:
        """
        返回当前设置图像的图像嵌入，形状为1xCxHxW，其中C是嵌入维度，(H,W)是SAM的嵌入空间维度（通常C=256, H=W=64）。
        """
        if not self.is_image_set:  # 检查图像是否已设置
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
                self.features is not None
        ), "Features must exist if an image has been set."  # 确保特征存在
        return self.features  # 返回图像特征

    @property
    def device(self) -> torch.device:  # 获取模型所用的设备（CPU或GPU）
        return self.model.device

    def reset_image(self) -> None:
        """重置当前设置的图像状态。"""
        self.is_image_set = False  # 设置图像未设置标志
        self.features = None  # 清除特征
        self.orig_h = None  # 清除原始高度
        self.orig_w = None  # 清除原始宽度
        self.input_h = None  # 清除输入高度
        self.input_w = None  # 清除输入宽度
