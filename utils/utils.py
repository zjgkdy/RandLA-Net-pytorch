import os
import yaml
import numpy as np
import open3d as o3d

def load_yaml(path):
    try:
        print(f"\033[32m Opening arch config file {path}\033[0m")
        yaml_data = yaml.safe_load(open(path, 'r'))
        return yaml_data
    except Exception as e:
        print(e)
        print(f"Error opening {path} yaml file.")
        quit()

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

def viz_pointcloud_with_labels(input_dict, learning_map_inv, color_map, root_path):
    seqs = input_dict['seq']
    filenames = input_dict['filename']
    points = input_dict["points"].cpu().numpy()[:, :, :3]
    labels = input_dict["labels"].cpu().numpy()

    B, N, D = points.shape
    for b in range(B):
        seq = seqs[b]
        filename = filenames[b]
        pc = points[b]
        label = labels[b]
        viz_color = np.zeros_like(pc)
        for idx in np.unique(label):
            color = np.array(color_map.get(learning_map_inv.get(int(idx), 0), [0, 0, 0])) / 255.0
            viz_color[label == idx] = color[::-1]

        # 构造 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(viz_color)
        os.makedirs(root_path, exist_ok=True) 
        save_path = os.path.join(root_path, seq+"_"+filename+".pcd")
        o3d.io.write_point_cloud(save_path, pcd)
