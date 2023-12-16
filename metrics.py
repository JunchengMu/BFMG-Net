import numpy as np
import torch
import torch.nn.functional as F


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


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def recall_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间
    output = torch.sigmoid(output)

    # 将输出和目标展平为一维数组
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例数目
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # 计算真实的正例数目
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))

    # 计算召回率
    recall = true_positives / (actual_positives + smooth)

    return recall


def precision_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间
    output = torch.sigmoid(output)

    # 将输出和目标展平为一维数组
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例数目
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # 计算预测的正例数目
    predicted_positives = np.sum(np.round(np.clip(output, 0, 1)))

    # 计算精确度
    precision = true_positives / (predicted_positives + smooth)

    return precision


def accuracy_s(output, target):
    # 将输出用sigmoid函数压缩到0到1之间，并四舍五入为0或1
    predicted = torch.round(torch.sigmoid(output))

    # 将预测值和目标值转换为CPU上的numpy数组
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算准确度
    accuracy = np.mean(predicted == target)

    return accuracy


def jaccard_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)
