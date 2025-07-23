# Common
import os
import yaml
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from dataset.semkitti_trainset import SemanticKITTI
from config.config import ConfigSemanticKITTI as cfg
from utils.metric import compute_acc, IoUCalculator
from network.RandLANet import Network
from network.loss_func import compute_loss
from utils.utils import compute_avg_grad_norm, get_learning_rate, reduce_value

torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', default='config/semantic-kitti.yaml', help='semantic-kitti.yaml path')
parser.add_argument('--dataset', default='./data/sequences_0.06', help='Dataset to train with. The parent directory of sequences. No Default.')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log/debug', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=20, help='Batch Size during validation [default: 30]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers [default: 5]')
parser.add_argument('--val_interval', type=int, default=5, help='Number of validation interval')
parser.add_argument('--sync_batchnorm', type=bool, default=False, help='Whether use sync batchnorm')
parser.add_argument('--local_rank', type=int, help='local rank for distributed training', default=0)
FLAGS = parser.parse_args()


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def init_distributed_mode(FLAGS):
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:  
        print('Using distributed mode')
        FLAGS.distributed = True
        FLAGS.rank = int(os.environ.get("RANK", 0))
        FLAGS.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        FLAGS.world_size = int(os.environ.get("WORLD_SIZE", 1))
        FLAGS.device = torch.device(f"cuda:{FLAGS.local_rank}")
        torch.cuda.set_device(FLAGS.local_rank)  #绑定当前进程到对应 GPU
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

class Trainer:
    def __init__(self):
        self.log_dir = FLAGS.log_dir
        self.val_interval = FLAGS.val_interval
       
        # get_dataset & dataloader
        DATA = yaml.safe_load(open(FLAGS.yaml_config, 'r'))
        self.train_dataset = SemanticKITTI('training', dataset_path=FLAGS.dataset, dataset_cfg=DATA)
        self.val_dataset  = SemanticKITTI('validation', dataset_path=FLAGS.dataset, dataset_cfg=DATA)
        
        # Distributed Train
        init_distributed_mode(FLAGS)

        # Dataloader
        if FLAGS.distributed:
            # 给每个rank对应的GPU进程分配训练的样本索引
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        else:
            # 非分布式训练，直接使用随机或顺序采样器
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        
        # 将样本索引组成batch
        self.train_batch_sampler = torch.utils.data.BatchSampler(self.train_sampler, FLAGS.batch_size, drop_last=True)
        self.val_batch_sampler = torch.utils.data.BatchSampler(self.val_sampler, FLAGS.batch_size, drop_last=False)
            
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler, 
            num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, # 设置 worker 的随机种子
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,    
            num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
        
        # Network & Optimizer
        self.net = Network(cfg)
        self.net = self.net.to(FLAGS.device)
        if FLAGS.distributed:
            if FLAGS.sync_batchnorm:
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(FLAGS.device)
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[FLAGS.local_rank],
                output_device=FLAGS.local_rank,
                find_unused_parameters=False  
            )
        else:
            self.net = self.net.to(FLAGS.device)
        
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"Total number of parameters: {total_params / 1e6:.2f}M")

        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        # Loss Function
        class_weights = torch.from_numpy(self.train_dataset.get_class_weight()).float().cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        
        # Tensorboard Writer and Logging
        if FLAGS.distributed and FLAGS.rank != 0:
            self.logger = None
            self.tf_writer = None
        else:
            os.makedirs(FLAGS.log_dir, exist_ok=True)
            logging.basicConfig(level=logging.DEBUG, 
                                format='%(asctime)s %(levelname)s: %(message)s', 
                                datefmt='%Y%m%d %H:%M:%S', 
                                filename=os.path.join(FLAGS.log_dir, 'log_train.txt'))
            self.logger = logging.getLogger("Trainer")       
            self.tf_writer = SummaryWriter(self.log_dir)

    def train_one_epoch(self):
        if FLAGS.distributed:
            self.train_sampler.set_epoch(self.cur_epoch)
            
        self.net.train()  # set model to training mode
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        for batch_idx, batch_data in enumerate(tqdm_loader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                elif type(batch_data[key]) is torch.Tensor:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()
            # Forward pass
            end_points = self.net(batch_data)
            loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)
            loss.backward()
            self.optimizer.step()
            
            # print info
            if batch_idx % 50 == 0 and self.tf_writer is not None and self.logger is not None:
                avg_grad_norm = compute_avg_grad_norm(self.net)
                current_lr = get_learning_rate(self.optimizer)
                print(f"[Epoch {self.cur_epoch :03d} | Iter {batch_idx:04d}] ",
                        f"Loss: {loss.item():.4f} | ",
                        f"GradNorm: {avg_grad_norm:.6f} | ",
                        f"LR: {current_lr:.6f}",)
                self.tf_writer.add_scalar('Train/Loss', loss.item(), self.cur_epoch * len(self.train_loader) + batch_idx)
                self.tf_writer.add_scalar('Train/GradNorm', avg_grad_norm, self.cur_epoch * len(self.train_loader) + batch_idx)
                self.tf_writer.add_scalar('Train/LR', current_lr, self.cur_epoch * len(self.train_loader) + batch_idx)
                
        self.scheduler.step()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            if self.logger is not None:
                self.logger.info(f'**** EPOCH {epoch:03d} ****')
                print(f'**** EPOCH {epoch:03d} ****')
                
            self.train_one_epoch()
            
            if epoch % self.val_interval == 0:
                if self.logger is not None:
                    self.logger.info(f'**** EVAL EPOCH {epoch:03d} ****')
                    print(f'**** EVAL EPOCH {epoch:03d} ****')
                    
                mean_iou = self.validate()                
                
                if self.logger is not None:
                    print(f"MeanIou = {mean_iou:.4f}, highest_val_iou = {self.highest_val_iou:.4f}")
                    # Save best checkpoint
                    if mean_iou > self.highest_val_iou:
                        self.highest_val_iou = mean_iou
                        checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
                        self.save_checkpoint(checkpoint_file)

    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg, FLAGS.distributed, FLAGS.device)

        val_loss_total = torch.zeros(1).to(FLAGS.device)
        val_acc_total = torch.zeros(1).to(FLAGS.device)
        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                end_points = self.net(batch_data)

                loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)
                val_loss_total += loss.detach()

                acc, end_points = compute_acc(end_points)
                val_acc_total += acc
                iou_calc.add_data(end_points)

        mean_loss = val_loss_total / len(self.val_loader)
        mean_acc = val_acc_total / len(self.val_loader)
        mean_iou, iou_list = iou_calc.compute_iou()
        
        if FLAGS.device != torch.device("cpu"):
            torch.cuda.synchronize(FLAGS.device)
        
        mean_loss = reduce_value(mean_loss, average=False)
        mean_acc = reduce_value(mean_acc, average=False)
        
        if self.logger is not None and self.tf_writer is not None:
            s = 'IoU:'
            for iou_tmp in iou_list:
                s += '{:5.2f} '.format(100 * iou_tmp)
            
            print(f"[Epoch {self.cur_epoch :03d}",
                            f"Mean_Loss: {mean_loss.item():.4f} | ",
                            f"Mean_Acc: {mean_acc.item():.6f} | ",)
            print(s)
            self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
            self.logger.info(s)
            self.tf_writer.add_scalar('Eval/mean_loss', mean_loss.item(), self.cur_epoch)
            self.tf_writer.add_scalar('Eval/mean_acc', mean_acc.item(), self.cur_epoch)
            self.tf_writer.add_scalar('Eval/mean_iou', mean_iou, self.cur_epoch)

        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
