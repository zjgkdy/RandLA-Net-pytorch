import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemanticKITTIDataset(Dataset):
    def __init__(self, root_dir, DATA, CFG, split="train", transform=None):
        self.color_map = DATA["color_map"]
        self.movable_learning_map = DATA["movable_learning_map"]
        self.movable_learning_map_inv = DATA["movable_learning_map_inv"]
        self.root_dir = root_dir
        self.train_sequences = DATA["split"]["train"]
        self.valid_sequences = DATA["split"]["valid"]
        self.ignore_index = DATA["ignore_index"]
        self.num_points = CFG.num_points
        self.split = split
        self.transform = transform

        self.point_paths = []
        self.label_paths = []

        if self.split == "train":
            data_sequences = self.train_sequences
        elif self.split == "valid":
            data_sequences = self.valid_sequences
             
        for seq in data_sequences:
                seq = f"{int(seq):02d}"
                scan_dir = os.path.join(root_dir, "sequences", seq, "velodyne")
                label_dir = os.path.join(root_dir, "sequences", seq, "labels")
                for fname in sorted(os.listdir(scan_dir)):
                    if fname.endswith(".bin"):
                        self.point_paths.append(os.path.join(scan_dir, fname))
                        self.label_paths.append(os.path.join(label_dir, fname.replace(".bin", ".label")))

    def random_sample(self, points, labels):
        N = points.shape[0]
        if N >= self.num_points :
            choice = np.random.choice(N, self.num_points , replace=False)
        else:
            choice = np.random.choice(N, self.num_points , replace=True)

        return points[choice], labels[choice]

    def __len__(self):
        return len(self.point_paths)

    def __getitem__(self, idx):
        # 加载点云
        point_path = self.point_paths[idx]
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        # 加载标签
        label_path = self.label_paths[idx]
        labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)

        # SemanticKITTI 标签是 32 位，前 16 位为类别，后 16 位为 instance id
        sem_labels = (labels & 0xFFFF).astype(np.int16)
        inst_labels = (labels >> 16).astype(np.int16)

        # 应用标签映射
        if self.movable_learning_map:
            sem_labels = np.vectorize(lambda x: self.movable_learning_map.get(x, self.ignore_index))(sem_labels)

        if self.transform:
            points, sem_labels = self.transform(points, sem_labels)

        # 随机采样固定点数：
        if self.split == "train":
            points, sem_labels = self.random_sample(points, sem_labels)

        return {
            'points': torch.from_numpy(points),       # (N, 4)
            'labels': torch.from_numpy(sem_labels)         # (N,)
        }
