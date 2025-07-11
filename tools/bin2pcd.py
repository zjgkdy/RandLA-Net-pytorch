import numpy as np
import open3d as o3d

if __name__ == "__main__":
    bin_path = "/home/luoteng/Code/RandLA-Net-pytorch/data/semantic_kitti/sequences/00/velodyne/000000.bin"
    label_path = "/home/luoteng/Code/RandLA-Net-pytorch/data/semantic_kitti/sequences/00/labels/000000.label"
    
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N, 4) = x,y,z,intensity
    xyz = point_cloud[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    o3d.io.write_point_cloud("000000_src.pcd", pcd)
    print("Saved to 000000_src.pcd")
