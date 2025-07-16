import numpy as np
import torch

def get_accuracy_tensor(scores, labels, device):
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions
    predictions = torch.max(scores, dim=-2).indices
    acc_tensor = torch.zeros((2, num_classes), dtype=torch.int64, device=device)
    accuracy_mask = predictions == labels
    for label in range(num_classes):            
        label_mask = labels == label
        acc_tensor[0][label] = (accuracy_mask & label_mask).float().sum()
        acc_tensor[1][label] = label_mask.float().sum()
    return acc_tensor

def get_iou_tensor(scores, labels, device):
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions
    predictions = torch.max(scores, dim=-2).indices
    iou_tensor = torch.zeros((2, num_classes), dtype=torch.int64, device=device)
    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou_tensor[0][label] = (pred_mask & labels_mask).float().sum()
        iou_tensor[1][label] = (pred_mask | labels_mask).float().sum()
    return iou_tensor
