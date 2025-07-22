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

