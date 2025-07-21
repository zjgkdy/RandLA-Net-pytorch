import os
import torch
import numpy as np
import open3d as o3d
import torch.distributed as dist

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
    
def reduce_value(value, average=True):
    world_size = int(os.environ.get('WORLD_SIZE', 1))    
    if world_size < 2:
        return value
    else:
        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= world_size
        return value

def create_pcd(points_xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    return pcd
