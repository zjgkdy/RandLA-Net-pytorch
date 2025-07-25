from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import importlib

class SemanticKITTI(torch_data.Dataset):
    def __init__(self, mode, dataset_path, dataset_cfg, model_cfg, data_list=None):
        self.name = 'SemanticKITTI'
        self.dataset_path = dataset_path
        self.raw_color_map = dataset_cfg["color_map"]
        self.learning_map_inv = dataset_cfg["learning_map_inv"]
        self.color_map = {k: self.raw_color_map[v] for k, v in self.learning_map_inv.items()}

        config_module = importlib.import_module(f'config.{model_cfg}')
        config_class = getattr(config_module, "ConfigSemanticKITTI")
        self.model_cfg = config_class()


        self.sampler = self.model_cfg.sampler
        self.num_points = self.model_cfg.num_points
        self.num_classes = self.model_cfg.num_classes
        self.ignored_labels = np.sort([0])

        self.mode = mode
        if data_list is None:
            if mode == 'training':
                seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            elif mode == 'validation':
                seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind, pc_path = self.spatially_regular_gen(item, self.data_list)
        return selected_pc, selected_labels, selected_idx, cloud_ind, pc_path

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
        
        if self.sampler == "crop_sampler":
            # crop a small point cloud
            pick_idx = np.random.choice(len(pc), 1)
            selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx) # 以 pc[pick_idx] 为中心裁剪局部区域
        elif self.sampler == "random_sampler":
            selected_pc, selected_labels, selected_idx = self.random_sample(pc, labels)
        return selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32), pc_path

    def random_sample(self, points, labels):
        N = points.shape[0]
        if N >= self.num_points :
            choice = np.random.choice(N, self.num_points , replace=False)
        else:
            choice = np.random.choice(N, self.num_points , replace=True)

        return points[choice], labels[choice], choice

    def get_data(self, file_path):
        """ Read points, labels and search_tree data.

        Args:
            file_path (_type_): File pll;ath

        Returns:
            _type_: _description_
        """
        seq_id = file_path[0]
        frame_id = file_path[1]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # load labels
        label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    def crop_pc(self, points, labels, search_tree, pick_idx):
        """裁剪一块局部区域，以pick_idx点为中心的

        Args:
            points (_type_): 原始点云
            labels (_type_): 原始点云标签
            search_tree (_type_): 搜索树
            pick_idx (_type_): 中心点索引

        Returns:
            _type_: _description_
        """
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=self.model_cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.model_cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, self.model_cfg.k_n) # 近邻点索引集
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.model_cfg.sub_sampling_ratio[i], :] # 降采样点集
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.model_cfg.sub_sampling_ratio[i], :] # 降采样近邻点索引集：降采样点对原始点
            up_i = DP.knn_search(sub_points, batch_pc, 1) # 上采样近邻点索引集：原始点对降采样点，用于上采样恢复特征
            input_points.append(batch_pc) # 输入点集
            input_neighbors.append(neighbour_idx) # 输入近邻点集
            input_pools.append(pool_i) # 降采样近邻点集
            input_up_samples.append(up_i) # 上采样近邻点集
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):
        selected_pc, selected_labels, selected_idx, cloud_ind, pc_path = [], [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            pc_path.append(batch[i][4])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = self.model_cfg.num_layers
        inputs = {}
        inputs['meta_info'] = {"pc_path": pc_path}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float()) # 当前层输入点云
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long()) # 输入点云近邻索引集
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long()) # 降采样近邻点索引集：降采样点对原始点
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long()) # 上采样近邻点索引集：原始点对降采样点，用于上采样恢复特征
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()

        return inputs
