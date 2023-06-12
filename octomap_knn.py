import matplotlib
matplotlib.use("tkagg")

import argparse
import os
import octomap
import numpy as np
import open3d as o3d

import yaml
import json
from tqdm import tqdm

# import utils
from range_image_map import segment_ground_multi_res
# import evaluation


import logging
logging.basicConfig(
    format='%(filename)s %(lineno)d %(asctime)s %(levelname)s - %(message)s', level=logging.INFO)

import evadd.io_utils as IO
from evadd.pc_utils import PC


def get_scan_wise_labels(octree: octomap.OcTree,
                         scan: o3d.geometry.PointCloud(),
                         scan_no: int,
                         nearest_neighbors: int,
                         ground_removal: bool,
                         args=None):
    """
       Provides final ground points calculated using certain
       resolution and after application of height filter

       Args:
           octree (octomap.OcTree): Octree created while generating static and dynamic map
           scan (o3d.geometry.PointCloud()): Point Cloud for which labels are required
           scan_no(int): Scan number for the particular scan
           nearest_neighbors(int): Number of nearest neighbors to be searched for labeling of unknown points
           ground_removal(bool): True -> To apply ground removal
                                 False -> For results without ground removal
    """

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

    colors = np.full((len(points), 3), [1.0, 0.0, 0.0])
    colors[empty_idx] = [0.0, 0.0, 1.0]

    scan.colors = o3d.utility.Vector3dVector(np.asarray(colors))

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

    if args.show:
        static_arr = np.asarray(scan.select_by_index(static_idx).points)
        dynamic_arr = np.asarray(scan.select_by_index(dynamic_idx).points)
        PC.show_pcs([static_arr, dynamic_arr], window_name="Static and dynamic")

    # Storing labels for input scan
    if ground_removal:
        file_name = f"{args.output_dir}/{str(scan_no).zfill(6)}"
    else:
        file_name = f"{args.output_dir}/orig_{str(scan_no).zfill(6)}"
    labels.reshape((-1)).astype(np.int32)
    np.save(file_name + '.npy', labels)


def shortest_distance(x1, y1, z1, a, b, c, d):
    """
    distance to plane
    """
    d = np.abs((a * x1 + b * y1 + c * z1 + d))
    e = (np.sqrt(a * a + b * b + c * c))
    dists = d/e
    return dists


def calc_dist(poses):
    """
    计算点之间的距离
    """
    distances_list = []
    all_pose = []
    last_ego_xyz = np.array([0.0, 0.0, 0.0]).T
    for p in poses:
        ego_xyz = p @ np.array([0.0, 0.0, 0.0, 1.0]).T
        ego_xyz = ego_xyz[:3].reshape(3, 1)
        all_pose.append(ego_xyz)
        distance = np.sqrt(np.sum(np.square(ego_xyz - last_ego_xyz)))
        last_ego_xyz = ego_xyz
        distances_list.append(distance)
    return distances_list

def main(args):
    # Loading Parameters
    with open(args.config) as f:
        config = yaml.safe_load(f)

    nearest_neighbors = config['Octomap']['nearest_neighbors']
    resolution = config['Octomap']['resolution']
    ground_removal = config['Octomap']['ground_removal']
    height_filter = config['Octomap']['height_filter']
    octree = octomap.OcTree(resolution)

    hm_resolution = config['Height_Map']['resolution']
    fwd_range = (config['Height_Map']['backward_range'], config['Height_Map']['fwd_range'])
    side_range = (config['Height_Map']['right_range'], config['Height_Map']['left_range'])
    height_range = (config['Height_Map']['bottom'], config['Height_Map']['top'])

    store_pcd = config['Results']['store_pcd']
    store_individual_label = config['Results']['store_individual_label']

    poses = IO.load_pose_to_SE3(args.pose_fpn)
    dists = calc_dist(poses)

    octree.clear()
    octree = octomap.OcTree(resolution)
    pcds = {}
    final_map = o3d.geometry.PointCloud()

    start_idx, end_idx = 0, len(poses) - 1
    max_dis = dists[end_idx]

    # load all pc
    for scan_no in range(start_idx, end_idx):
        pcd_fpn = f"{args.pcd_dir}/{str(scan_no).zfill(6)}.pcd"
        pcd = o3d.io.read_point_cloud(pcd_fpn)
        pcds[scan_no] = pcd

        final_map = final_map + pcd.transform(poses[scan_no])

    # load or create octo map
    octo_map_file = f"{args.output_dir}/octo_map_res{resolution}.bt"
    if os.path.exists(octo_map_file):
        octree.readBinary(octo_map_file.encode())
        logging.info(f"Read from map {octo_map_file}.")
    else:
        for scan_no in range(start_idx, end_idx):

            pcd = pcds[scan_no]

            # Applying ground segmentation on Point Cloud
            # ground_pcd, non_ground_pcd, ground_indices, non_ground_indices = \
            #     segment_ground_multi_res(pcd=pcd,
            #                              res=hm_resolution,
            #                              fwd_range=fwd_range,
            #                              side_range=side_range,
            #                              height_range=height_range,
            #                              height_threshold=-1.1)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.15,
                                            ransac_n=3,
                                            num_iterations=100)
            # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            ground_pcd = pcd.select_by_index(inliers)
            non_ground_pcd = pcd.select_by_index(inliers, invert=True)
            PC.show_pcs([np.asarray(ground_pcd.points), np.asarray(non_ground_pcd.points)], window_name="Ground and Non ground", args=args)

            points = np.asarray(pcd.points)
            ground_pcd = ground_pcd.transform(poses[scan_no])
            ground_points = np.asarray(ground_pcd.points)

            pose = poses[scan_no]

            # Inserting Point Cloud into OcTree
            octree.insertPointCloud(
                pointcloud=points,
                origin=np.array([pose[0][3], pose[1][3], pose[2][3]], dtype=float),
                maxrange=80.0,
            )
            logging.info(f"Add {pcd_fpn} to octree.")

            # Setting ground points as static in the Occupancy grid
            if ground_removal:
                for pt in range(len(ground_points)):
                    valid, key = octree.coordToKeyChecked(ground_points[pt])
                    if valid:
                        node = octree.search(key)
                        try:
                            node.setValue(200.0)
                        except octomap.NullPointerException:
                            pass
                logging.info("Set ground static.")

            # Setting points above a certain height as static in the Occupancy grid
            if height_filter:
                height_threshold = 4.5
                [a, b, c, d] = plane_model
                height = shortest_distance(points[:, 0], points[:, 1], points[:, 2], \
                        a, b, c, d)
                mask = height > height_threshold
                ht_points = points[mask]
                for pt in range(len(ht_points)):
                    valid, key = octree.coordToKeyChecked(ht_points[pt])
                    if valid:
                        node = octree.search(key)
                        try:
                            node.setValue(200.0)
                        except octomap.NullPointerException:
                            pass
                logging.info("超高的点，设置为静态.")

            octree.updateInnerOccupancy()
        octo_map_file.writeBinary(octo_map_file.encode())
    
        logging.info(f"Write to map {octo_map_file}.")

    # Extracting labels from OcTree
    final_points = np.asarray(final_map.points)
    labels = octree.getLabels(final_points)

    occupied_idx, empty_idx, unknown_idx = [], [], []
    for i in range(len(labels)):
        if labels[i] == 1:
            occupied_idx.append(i)
        elif labels[i] == 0:
            empty_idx.append(i)
        else :
            unknown_idx.append(i)
    logging.info("check occupied or not")

    pcd_static = final_map.select_by_index(occupied_idx)
    pcd_dynamic = final_map.select_by_index(empty_idx)

    color_static = np.full((len(np.asarray(pcd_static.points)), 3), [1.0, 0.0, 0.0])
    color_dynamic = np.full((len(np.asarray(pcd_dynamic.points)), 3), [0.0, 0.0, 1.0])

    pcd_static.colors = o3d.utility.Vector3dVector(np.asarray(color_static))
    pcd_dynamic.colors = o3d.utility.Vector3dVector(np.asarray(color_dynamic))

    pcd = pcd_static + pcd_dynamic
    pred_tree = o3d.geometry.KDTreeFlann(pcd)
    color = np.asarray(pcd.colors)

    # Assigning labels to unknown labels
    for pt in unknown_idx:
        [_, idx, _] = pred_tree.search_knn_vector_3d(final_points[pt], nearest_neighbors)
        final_score = np.mean(color[idx, 0])

        if final_score > 0.5:
            occupied_idx.append(pt)
        else:
            empty_idx.append(pt)
    logging.info("用knn区分unknown")

    pcd_static = final_map.select_by_index(occupied_idx)
    pcd_dynamic = final_map.select_by_index(empty_idx)

    # Downsampling static and dynamic point cloud
    data = pcd_static.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                    [max_dis, max_dis, max_dis])
    pcd_static = data[0]
    data = pcd_dynamic.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                    [max_dis, max_dis, max_dis])

    pcd_dynamic = data[0]
    data = final_map.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                    [max_dis, max_dis, max_dis])
    down_idx = data[1]
    down_idx = down_idx[down_idx > 0]

    # Calculating static and dynamic indices for original map
    static_idx = list(set(occupied_idx) & set(down_idx))
    dynamic_idx = list(set(empty_idx) & set(down_idx))
    logging.info("计算动静index")

    # # Performing evaluation
    # print("performing eval ...")
    # ts, td, total_static, total_dynamic = evaluation.eval(static_idx=static_idx,
    #                                                       dynamic_idx=dynamic_idx,
    #                                                       start=start,
    #                                                       end=end,
    #                                                       poses=poses,
    #                                                       path_to_scan=path_to_scans,
    #                                                       path_to_gt=None)
    # print("TS", ts, "Total Static", total_static)
    # print("TD", td, "Total Dynamic", total_dynamic)

    # key = str(start) + "-" + str(end)
    # res = {key: {"TS": ts, "TD": td, "Total Voxels": total_static, "Total Dynamic Points": total_dynamic,
    #              "Accuracy": (ts + td) / (total_dynamic + total_static + 1e-8),
    #              "Accuracy_1": (ts / (2 * total_static + 1e-8)) + (td / (2 * total_dynamic + 1e-8)),
    #              "Recall": td / (total_dynamic + 1e-8)}}

    # # Storing results
    # json_data = json.dumps(res)
    # if ground_removal:
    #     f = open(f'./json/{seq}.json', 'a+')
    #     f.write(json_data + "\n")
    #     f.close()
    # else:
    #     f = open(f'./json/{seq}_orig.json', 'a+')
    #     f.write(json_data + "\n")
    #     f.close()

    # Storing static and dynamic point cloud for individual scans
    if store_pcd:
        o3d.io.write_point_cloud(f"{args.output_dir}/static.pcd", pcd_static)
        o3d.io.write_point_cloud(f"{args.output_dir}/dynamic.pcd", pcd_dynamic)

    # Storing labels for individual scans
    if store_individual_label:
        print("Storing Scan Wise Labels")
        for scan_no in tqdm(range(start_idx, end_idx)):
            pcd = pcds[scan_no]
            get_scan_wise_labels(octree, pcd, scan_no, nearest_neighbors, ground_removal,args)


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
