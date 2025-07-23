import numpy as np
import open3d as o3d

# SemanticKITTI Label ID 到 RGB颜色（BGR → RGB）
label_color_map = {
    0:  [0, 0, 0],
    1:  [255, 0, 0],
    10: [100, 150, 245],
    11: [100, 230, 245],
    13: [100, 80, 250],
    15: [30, 60, 150],
    16: [0, 0, 255],
    18: [80, 30, 180],
    20: [0, 0, 255],
    30: [255, 30, 30],
    31: [255, 40, 200],
    32: [150, 30, 90],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [175, 0, 75],
    50: [255, 200, 0],
    51: [255, 120, 50],
    52: [255, 150, 0],
    60: [150, 255, 170],
    70: [0, 175, 0],
    71: [135, 60, 0],
    72: [150, 240, 80],
    80: [255, 240, 150],
    81: [255, 0, 0],
    99: [50, 255, 255],
    252: [100, 150, 245],
    253: [255, 40, 200],
    254: [255, 30, 30],
    255: [150, 30, 90],
    256: [0, 0, 255],
    257: [100, 80, 250],
    258: [80, 30, 180],
    259: [0, 0, 255],
}

ignore_ids = {0, 1, 99}  # 可忽略的类别


def visualize_semantickitti_open3d(bin_path, label_path):
    # === 1. 加载点云 ===
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    # === 2. 加载标签 ===
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF

    # === 3. 过滤 ignore 类别 ===
    valid_mask = ~np.isin(semantic_labels, list(ignore_ids))
    xyz = xyz[valid_mask]
    semantic_labels = semantic_labels[valid_mask]

    # === 4. 映射颜色 ===
    color_map = np.array([
        label_color_map.get(label, [0, 0, 0]) for label in semantic_labels
    ], dtype=np.float32) / 255.0

    # === 5. 创建 Open3D 点云对象 ===
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color_map)

    # === 6. 显示 ===
    o3d.io.write_point_cloud("./viz.pcd", pcd)
    o3d.visualization.draw_geometries([pcd], window_name="SemanticKITTI - Open3D")

if __name__ == "__main__":
    # 修改为你本地的数据路径
    bin_file = "/home/luoteng/dataset/SemanticKITTI/dataset/sequences/00/velodyne/000239.bin"
    label_file = "/home/luoteng/dataset/SemanticKITTI/dataset/sequences/00/labels/000239.label"

    visualize_semantickitti_open3d(bin_file, label_file)