import torch
import numpy as np
import threading
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from utils.utils import reduce_value

class IoUCalculator:
    def __init__(self, cfg, distributed=False, device="cpu"):
        self.distributed = distributed
        self.device = device
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg
        self.lock = threading.Lock()

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.cfg.num_classes, 1))
        conf_tensor = torch.tensor(conf_matrix, device=self.device)
        conf_tensor = reduce_value(conf_tensor, average=False)
        
        self.lock.acquire()
        conf_matrix_sync = conf_tensor.cpu().numpy()
        self.gt_classes += np.sum(conf_matrix_sync, axis=1)
        self.positive_classes += np.sum(conf_matrix_sync, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix_sync)
        self.lock.release()

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / \
                    float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points
