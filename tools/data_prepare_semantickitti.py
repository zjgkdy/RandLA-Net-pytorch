import os
import yaml
import pickle
import argparse
import functools
import numpy as np
from os.path import join, exists
from sklearn.neighbors import KDTree
from utils.data_process import DataProcessing as DP
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', default=None, help='source dataset path [default: None]')
parser.add_argument('--dst_path', default=None, help='destination dataset path [default: None]')
parser.add_argument('--grid_size', type=float, default=0.06, help='Subsample Grid Size [default: 0.06]')
parser.add_argument('--yaml_config', default='config/semantic-kitti-movable.yaml', help='semantic-kitti.yaml path')
FLAGS = parser.parse_args()


data_config = FLAGS.yaml_config
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    
def process_scan(seq_id, scan_id, FLAGS, remap_lut):
    print(seq_id, "\t", scan_id)
    seq_path = join(FLAGS.src_path, seq_id)
    seq_path_out = join(FLAGS.dst_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    os.makedirs(pc_path_out, exist_ok=True)
    os.makedirs(KDTree_path_out, exist_ok=True)

    points = DP.load_pc_kitti(join(pc_path, scan_id))

    if int(seq_id) < 11:
        label_path = join(seq_path, 'labels')
        label_path_out = join(seq_path_out, 'labels')
        os.makedirs(label_path_out, exist_ok=True)
        labels = DP.load_label_kitti(join(label_path, scan_id[:-4] + '.label'), remap_lut)
        sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=FLAGS.grid_size)
        np.save(join(pc_path_out, scan_id)[:-4], sub_points)
        np.save(join(label_path_out, scan_id)[:-4], sub_labels)
    else:
        sub_points = DP.grid_sub_sampling(points, grid_size=FLAGS.grid_size)
        np.save(join(pc_path_out, scan_id)[:-4], sub_points)

    # KDTree and projection
    search_tree = KDTree(sub_points)
    with open(join(KDTree_path_out, scan_id[:-4] + '.pkl'), 'wb') as f:
        pickle.dump(search_tree, f)
                
def main():
    dataset_path = FLAGS.src_path
    output_path = FLAGS.dst_path
    seq_list = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    
    for seq_id in seq_list:        
        if int(seq_id) >= 11:
            continue
        print(f"Processing sequence {seq_id}")
        seq_path_out = join(output_path, seq_id)
        os.makedirs(seq_path_out, exist_ok=True)

        pc_path = join(dataset_path, seq_id, 'velodyne')
        scan_list = np.sort(os.listdir(pc_path))

        # 并行处理每个 scan
        with ProcessPoolExecutor(max_workers=10) as executor:  # 可根据 CPU 调整
            executor.map(functools.partial(process_scan, seq_id, FLAGS=FLAGS, remap_lut=remap_lut), scan_list)
            
if __name__ == "__main__":
    main()