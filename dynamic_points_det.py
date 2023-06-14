import matplotlib
matplotlib.use("tkagg")

import argparse
import os
import octomap
import numpy as np
import open3d as o3d
import glob
import yaml

from range_image_map import segment_ground_multi_res


import logging
logging.basicConfig(
    format='%(filename)s %(lineno)d %(asctime)s %(levelname)s - %(message)s', level=logging.INFO)

import evadd.io_utils as IO
from evadd.pc_utils import PC


def main(args):
    # Loading Parameters
    with open(args.config) as f:
        config = yaml.safe_load(f)

    nearest_neighbors = config['Octomap']['nearest_neighbors']
    resolution = config['Octomap']['resolution']
    ground_removal = config['Octomap']['ground_removal']
    height_filter = config['Octomap']['height_filter']

    poses = IO.load_pose_to_SE3(args.pose_fpn)

    octree = octomap.OcTree(resolution)

    # load or create octo map
    octo_map_file = f"{args.output_dir}/octo_map_res{resolution}.bt"
    octree.readBinary(octo_map_file.encode())
    logging.info(f"Read from map {octo_map_file}.")

    hm_resolution = config['Height_Map']['resolution']
    fwd_range = (config['Height_Map']['backward_range'], config['Height_Map']['fwd_range'])
    side_range = (config['Height_Map']['right_range'], config['Height_Map']['left_range'])
    height_range = (config['Height_Map']['bottom'], config['Height_Map']['top'])

    pcds = glob.glob(args.pcd_dir + "/*.pcd")

    # load all pc
    for pcd_fpn in pcds:
        pcd = o3d.io.read_point_cloud(pcd_fpn)
        
        base_name, ext = os.path.splitext(os.path.basename(pcd_fpn))
        scan_no = int(base_name)
        # get_scan_wise_labels(octree, pcd.transform(poses[scan_no]), scan_no, nearest_neighbors, ground_removal,args)
        scan = pcd.transform(poses[scan_no])

        # Extracting labels from OcTree for input scan
        points = np.asarray(scan.points)
        labels = octree.getLabels(points)

        empty_idx, occupied_idx, unknown_idx = [], [], []

        for i in range(len(labels)):
            if labels[i] == 0:
                empty_idx.append(i)
            elif labels[i] == 1:
                occupied_idx.append(i)
            else:
                unknown_idx.append(i)

        # colors = np.full((len(points), 3), [1.0, 0.0, 0.0])
        # colors[empty_idx] = [0.0, 0.0, 1.0]
        # scan.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        if False:
            known_idx = np.concatenate((occupied_idx, empty_idx), axis=None)
            pcd_known = scan.select_by_index(known_idx)
            pred_tree = o3d.geometry.KDTreeFlann(pcd_known)
            color = np.asarray(pcd_known.colors)
            static_idx, dynamic_idx = [], []
            # Assigning labels to unknown labels
            for pt in unknown_idx:
                [_, idx, _] = pred_tree.search_knn_vector_3d(points[pt], nearest_neighbors)

                final_score = np.mean(color[idx, 0])

                if final_score > 0.5:
                    static_idx.append(pt)
                else:
                    dynamic_idx.append(pt)

            static_idx = np.concatenate((occupied_idx, static_idx), axis=None)
            dynamic_idx = np.concatenate((empty_idx, dynamic_idx), axis=None)
            static_idx = static_idx.astype(np.int32)
            dynamic_idx = dynamic_idx.astype(np.int32)
            labels = np.full((len(static_idx) + len(dynamic_idx),), 9)
            labels[dynamic_idx] = 251
        else:
            labels = np.full((len(labels),), -1)
            labels[occupied_idx] = 9
            labels[empty_idx] = 251

        if False:
            ground_pcd, non_ground_pcd, ground_indices, non_ground_indices = \
                segment_ground_multi_res(pcd=scan,
                                            res=hm_resolution,
                                            fwd_range=fwd_range,
                                            side_range=side_range,
                                            height_range=height_range,
                                            height_threshold=-1.1)
        else:
            plane_model, ground_indices = pcd.segment_plane(distance_threshold=0.3,
                                            ransac_n=3,
                                            num_iterations=200)
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        # ground_pcd = scan.select_by_index(ground_indices)
        # non_ground_pcd = scan.select_by_index(ground_indices, invert=True)

        occupied_idx = np.union1d(np.array(occupied_idx).astype(np.int32), ground_indices)
        empty_idx = np.setdiff1d(np.array(empty_idx).astype(np.int32), ground_indices)

        if args.show:
            static_arr = np.asarray(scan.select_by_index(occupied_idx).points)
            dynamic_arr = np.asarray(scan.select_by_index(empty_idx).points)
            unknown_arr = np.asarray(scan.select_by_index(unknown_idx).points)
            PC.show_pcs([static_arr, dynamic_arr, unknown_arr], window_name="Static dynamic unknown")

        labels = np.full((len(labels),), -1)
        labels[occupied_idx.tolist()] = 9
        labels[empty_idx.tolist()] = 251

        # Storing labels for input scan
        if ground_removal:
            file_name = f"{args.output_dir}/{str(scan_no).zfill(6)}"
        else:
            file_name = f"{args.output_dir}/orig_{str(scan_no).zfill(6)}"
        labels.reshape((-1)).astype(np.int32)
        np.save(file_name + '.npy', labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--show', action="store_true", default=False)
    parser.add_argument('--vis', action="store_true", default=False)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--pcd_dir', default="")
    parser.add_argument('--pose_fpn', default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
