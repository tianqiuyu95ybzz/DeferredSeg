import numpy as np
import torch
import torch.nn.functional as F
'''
评估指标
'''
# 计算平均IoU
def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2#标签中有两个对象（或类别）
        pred_objects = 2

        # 计算标签和预测值之间的交集矩阵
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # 计算标签和预测值的面积
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)#扩展后的结果是一个列向量
        area_pred = np.expand_dims(area_pred, 0)#扩展后的结果是一个行向量

        # 计算并集
        union = area_true + area_pred - intersection

        # 排除背景的分析
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9 #避免除以0的情况发生

        # 计算IoU
        iou = intersection / union

        # 辅助函数：计算给定阈值下的精确度
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # 真正例 tp、假正例 fp 和假负例 fn
            false_positives = np.sum(matches, axis=0) == 0
            false_negatives = np.sum(matches, axis=1) == 0
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # 遍历IoU阈值
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0


# 计算批量样本的IoU
def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)

# 计算平均IoU的函数
def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)

# 计算IoU得分的函数
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def calculate_iou_with_mask(output, target, mask):
    """
    计算在指定掩码范围内的 IoU
    :param output: 模型预测结果 (batch_size, n_classes, H, W)
    :param target: 实际目标标签 (batch_size, n_classes, H, W)
    :param mask: 布尔掩码，用于选择需要计算 IoU 的区域 (batch_size, n_classes, H, W)
    :return: 在指定掩码范围内的 IoU 值
    """
    smooth = 1e-5

    # 检查维度是否一致
    if output.ndim != 4 or target.ndim != 4 or mask.ndim != 4:
        raise ValueError(f"Expected 4D inputs for output, target, and mask, but got shapes: {output.shape}, {target.shape}, {mask.shape}")

    # 将预测和目标都转换为 numpy 数组
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()  # 将输出从 GPU 转移到 CPU 并转换为 numpy
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()  # 同样将目标从 GPU 转移到 CPU
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()  # 将掩码也从 GPU 转移到 CPU 并转换为 numpy

    # 应用掩码，仅保留指定区域的输出和目标
    output_masked = output[mask == 1]  # 只在掩码为 True 的地方取值
    target_masked = target[mask == 1]  # 只在掩码为 True 的地方取值

    # 将输出和目标转换为二进制（即大于 0.5 为正类）
    output_ = output_masked > 0.5
    target_ = target_masked > 0.5

    # 计算交集和并集
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    # 返回 IoU 值
    return (intersection + smooth) / (union + smooth)



# 计算Dice系数
def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

# 计算准确率
def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)

# 计算阳性预测值
def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

# 计算敏感性
def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

def calculate_iou(prediction, ground_truth):
    intersection = prediction * ground_truth
    union = prediction+ground_truth-intersection
    iou=torch.sum(intersection)/torch.sum(union)if torch.sum(union) != 0 else torch.tensor(0.0)
    return iou
def calculate_dice(prediction, ground_truth):
    intersection =  torch.sum(prediction * ground_truth)
    dice = (2. * intersection) / ( torch.sum(prediction) + torch.sum(ground_truth) + 1e-8)
    return dice