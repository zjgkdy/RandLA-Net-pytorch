import numpy as np

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def compute_avg_grad_norm(model):
    total_norm = 0.0
    num_params = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1
    if num_params > 0:
        return (total_norm / num_params) ** 0.5
    else:
        return 0.0