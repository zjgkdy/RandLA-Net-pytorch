#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from utils.semkitti_vis.laserscan import SemLaserScan
from utils.semkitti_vis.laserscanvis import LaserScanVis

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument( 
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="08",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)
    print("*" * 80)

    # open config file
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # does sequence folder exist?
    pred_label_paths = os.path.join(FLAGS.dataset, FLAGS.sequence, "labels/pred")
    gt_label_paths = os.path.join(FLAGS.dataset, FLAGS.sequence, "labels/gt")
    if os.path.isdir(pred_label_paths):
        print("Predicted labels folder exists! Using labels from %s" % pred_label_paths)
    else:
        print("Predicted labels folder doesn't exist! Exiting...")
        quit()
    if os.path.isdir(gt_label_paths):
        print("Groundtruth labels folder exists! Using labels from %s" % gt_label_paths)
    else:
        print("Groundtruth labels folder doesn't exist! Exiting...")
        quit()
        
    # populate the pointclouds
    pred_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_label_paths)) for f in fn]
    pred_label_names.sort()
    gt_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(gt_label_paths)) for f in fn]
    gt_label_names.sort()

    color_dict =  {k: CFG["color_map"][v] for k, v in CFG["learning_map_inv"].items()}
    
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

    vis = LaserScanVis(scan=scan,
                       scan_names=scan_names,
                       pred_label_names=pred_label_names,
                       gt_label_names=gt_label_names,
                       offset=0,
                       semantics=True, instances=False)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
