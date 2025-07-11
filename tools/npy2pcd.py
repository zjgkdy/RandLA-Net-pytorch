import numpy as np
import open3d as o3d

if __name__ == "__main__":
    pc_path = "/home/luoteng/Code/RandLA-Net-pytorch/data/semantic_kitti/sequences_0.06/00/velodyne/000000.npy"
    label_path = "/home/luoteng/Code/RandLA-Net-pytorch/data/semantic_kitti/sequences_0.06/00/labels/000000.npy"
    
    points = np.load(pc_path)
    label = np.load(label_path)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # 保存为 .pcd 文件
    o3d.io.write_point_cloud("000000_dst.pcd", pcd)
    print("Saved to 000000_dst.pcd")