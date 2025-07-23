import os
import logging
import warnings
import argparse
import yaml
import numpy as np
import open3d as o3d
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from dataset.semkitti_testset import SemanticKITTI
from config.config import ConfigSemanticKITTI as cfg
from utils.metric import IoUCalculator, compute_acc

from network.RandLANet import Network
from utils.utils import reduce_value, create_pcd

torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', default='config/semantic-kitti.yaml', help='semantic-kitti.yaml path')
parser.add_argument('--dataset', default='./data/sequences_0.06', help='Dataset to evaluate with. The parent directory of sequences.')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--result_dir', default='result/', help='Dump dir to save prediction [default: result/]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during inference [default: 4]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers [default: 5]')
parser.add_argument('--local_rank', type=int, help='local rank for distributed training', default=0)
FLAGS = parser.parse_args()


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def init_distributed_mode(FLAGS):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print('Using distributed mode')
        FLAGS.distributed = True
        FLAGS.rank = int(os.environ.get("RANK", 0))
        FLAGS.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        FLAGS.world_size = int(os.environ.get("WORLD_SIZE", 1))
        FLAGS.device = torch.device(f"cuda:{FLAGS.local_rank}")
        torch.cuda.set_device(FLAGS.local_rank)  # Bind process to corresponding GPU
        dist.init_process_group(backend='nccl', init_method='env://', world_size=FLAGS.world_size, rank=FLAGS.rank)
        
        if FLAGS.rank == 0:
            print(f"Running on {FLAGS.world_size} GPUs")
    else:
        print('Not using distributed mode')
        FLAGS.distributed = False
        FLAGS.rank = 0
        FLAGS.world_size = 1
        FLAGS.local_rank = 0
        FLAGS.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return


class Inference:
    def __init__(self):       
        # Distributed setup
        init_distributed_mode(FLAGS)

        # Dataloader
        print("Opening data config file %s" % FLAGS.yaml_config)
        DATA = yaml.safe_load(open(FLAGS.yaml_config, 'r'))
        self.test_dataset = SemanticKITTI('validation', dataset_path=FLAGS.dataset, dataset_cfg=DATA)
        if FLAGS.distributed:
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.val_sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        
        self.val_batch_sampler = torch.utils.data.BatchSampler(self.val_sampler, FLAGS.batch_size, drop_last=False)
            
        self.val_loader = DataLoader(
            self.test_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True
        )
        
        # Network
        self.net = Network(cfg)
        self.net = self.net.to(FLAGS.device)
        
        # Load model checkpoint
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {CHECKPOINT_PATH}")
        else:
            raise ValueError(f"Checkpoint not found at {CHECKPOINT_PATH}")
            
    def run_inference(self):
        self.net.eval() # Set the model to evaluation mode
        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    elif type(batch_data[key]) is torch.Tensor:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                end_points = self.net(batch_data)
                path_infos = end_points["meta_info"]["pc_path"]
                points_xyz = end_points["xyz"]
                pre_labels = end_points["logits"].max(dim=1)[1] + 1 # 1~20
                gt_labels = end_points["labels"]
                valid_mask = gt_labels != 0
                for idx in range(len(path_infos)):
                    seq = path_infos[idx][0]
                    index = path_infos[idx][1]
                    xyz = points_xyz[0][idx].cpu().numpy()
                    pred_sem = pre_labels[idx].cpu().numpy()
                    gt_sem = gt_labels[idx].cpu().numpy()
                    assert xyz.shape[0] == pred_sem.shape[0] and pred_sem.shape[0] == gt_sem.shape[0]
                    
                    # Save sample points
                    save_points_dir = os.path.join(FLAGS.result_dir, seq+"/velodyne")
                    os.makedirs(save_points_dir, exist_ok=True)
                    points_path =  os.path.join(save_points_dir, index+".npz")
                    np.savez(points_path, xyz)
                    
                    # Save label files
                    save_label_dir = os.path.join(FLAGS.result_dir, seq+"/labels")
                    save_pred_label_dir = os.path.join(save_label_dir, "pred")
                    save_gt_label_dir = os.path.join(save_label_dir, "gt")
                    os.makedirs(save_pred_label_dir, exist_ok=True)
                    os.makedirs(save_gt_label_dir, exist_ok=True)
                    pred_label_path = os.path.join(save_pred_label_dir, index+".npz")
                    gt_label_path = os.path.join(save_gt_label_dir, index+".npz")
                    np.savez(pred_label_path, pred_sem)
                    np.savez(gt_label_path, gt_sem)
                    
                    # Save pcds
                    if idx % 20 == 0:
                        color_map = self.test_dataset.color_map
                        save_pcd_dir = os.path.join(FLAGS.result_dir, seq+"/pcd")
                        os.makedirs(save_pcd_dir, exist_ok=True)
                        pred_pcd_path = os.path.join(save_pcd_dir, index+"_pred.ply")
                        gt_pcd_path = os.path.join(save_pcd_dir, index+"_gt.ply")
                        pred_color = np.array([color_map[sem] for sem in pred_sem]) / 255.0
                        gt_color = np.array([color_map[sem] for sem in gt_sem]) / 255.0
                        pred_pcd = create_pcd(xyz, pred_color)
                        gt_pcd = create_pcd(xyz, gt_color)
                        o3d.io.write_point_cloud(pred_pcd_path, pred_pcd)
                        o3d.io.write_point_cloud(gt_pcd_path, gt_pcd)
                               
            if FLAGS.device != torch.device("cpu"):
                torch.cuda.synchronize(FLAGS.device)


def main():
    inference = Inference()
    inference.run_inference()

if __name__ == '__main__':
    main()


