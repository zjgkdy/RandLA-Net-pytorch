## RandLA-Net 源码阅读笔记
### 1. 原始点云采样方法
使用的并不是论文中的全局随机采样，而是随机裁剪一块局部区域，只能覆盖点云局部场景:
```
@staticmethod
def crop_pc(points, labels, search_tree, pick_idx):
    # crop a fixed size point cloud for training
    center_point = points[pick_idx, :].reshape(1, -1)
    select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
    select_idx = DP.shuffle_idx(select_idx)
    select_points = points[select_idx]
    select_labels = labels[select_idx]
    return select_points, select_labels, select_idx
```
数据预处理生成的KDTree只是用来提取局部区域点云，之后模型中的近邻搜索使用 nearest_neighbors 库在线生成KDTree， 因此将随机裁剪区域采样点云更换成全局随机采样策略，并不会影响模型的训练流程和速度。

### 2. 模型预测
模型输出逐点属于每个有效语义类别的概率，不输出属于忽略语义类别的概率，计算损失时对标签做了重映射来忽略掉无效的语义类别。

### 3. 输入数据字段解析
+ "xyz": 当前层输入点集 [N1, 3]
+ "neigh_idx": 输入点云近邻索引集 [N1, K]
+ "sub_idx": 降采样点云近邻索引集：降采样点在原始点云中的近邻 [N2, K]
+ "interp_idx": 上采样点云近邻索引集：原始点在降采样点云中的近邻，用于上采样恢复特征 [N1, 1]

### 4. 上采样和下采样
+ **下采样**：使用K个近邻点做MaxPooling得到下采样点的特征
+ **上采样**：使用最近邻点的特征

### 5. 分布式训练
1. 初始化分布式训练模型
```
def init_distributed_mode(FLAGS):
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:  
        print('Using distributed mode')
        FLAGS.distributed = True
        FLAGS.rank = int(os.environ.get("RANK", 0)) # 全局RANK
        FLAGS.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本机RANK
        FLAGS.world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总卡数
        FLAGS.device = torch.device(f"cuda:{FLAGS.local_rank}")
        torch.cuda.set_device(FLAGS.local_rank)  #绑定当前进程到对应 GPU
        dist.init_process_group(
            backend='nccl', 
            init_method='env://', 
            world_size=FLAGS.world_size, 
            rank=FLAGS.rank
        )
        
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
```
### 2. 数据集配置
```
if FLAGS.distributed:
    # 给每个rank对应的GPU进程分配训练的样本索引
    self.train_sampler = torch.utils.data.distributed.DistributedSample(
        self.train_dataset, 
        shuffle=True
    )
    self.val_sampler = torch.utils.data.distributed.DistributedSampler(
        self.val_dataset, 
        shuffle=False
    )
else:
    # 非分布式训练，直接使用随机或顺序采样器
    self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
    self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

# 将样本索引组成batch
self.train_batch_sampler = torch.utils.data.BatchSampler(
    self.train_sampler,
    FLAGS.batch_size,
    drop_last=True
)
self.val_batch_sampler = torch.utils.data.BatchSampler(
    self.val_sampler, 
    FLAGS.batch_size, 
    drop_last=False
)
    
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
```
### 3. 模型配置
```
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
```
### 4. 模型训练重采样
```
def train_one_epoch(self):
    if FLAGS.distributed:
        self.train_sampler.set_epoch(self.cur_epoch)
    .....
```
### 5. 多卡数据同步
```
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
```