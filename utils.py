import os
import copy
import sys

import numpy as np
import h5py
import trimesh
import open3d as o3d
import torch
from scipy.spatial import KDTree
from pytorch3d.loss import chamfer_distance

DATA_PATH = 'data'


def occ_vis_split(points):
    '''
    Splits pointcloud into visible and occluded sub-pointclouds from viewpoint of camera.
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100
    _, vis = pcd.hidden_point_removal(camera, radius)
    points = np.asarray(pcd.points)
    vis_pcd = points[vis]
    occ_pcd = np.delete(points, vis, axis=0)
    return vis_pcd, occ_pcd


def get_points_sdf_sample():
    '''
    Retrieves sdf sample points.

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    sdf_h5_path = os.path.join(DATA_PATH, 'ori_sample.h5')
    ori_pt, ori_sdf_val, input_points, input_sdfs, norm_params, \
        sdf_params  = get_sdf_h5(sdf_h5_path)
    input_points, input_sdfs = sample_points(input_points,input_sdfs)

    # Rotate points for 3-DOF VC
    hvc_metadata_path = os.path.join(DATA_PATH, 'hard_vc_metadata.txt')
    hvc_meta = np.loadtxt(hvc_metadata_path)
    hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
    input_points = apply_rotate(input_points, hvc_rotate_dict)

    metadata_path = os.path.join(DATA_PATH, 'metadata.txt')
    meta = np.loadtxt(metadata_path)
    rotate_dict = {'elev': meta[10][1], \
        'azim': meta[10][0]-180}

    input_points = apply_rotate(input_points, rotate_dict)
    return input_points, input_sdfs


def get_pointcloud_sample():
    '''
    Get pointcloud sample for __getitem__ during testing

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    pointcld_path = os.path.join(DATA_PATH, 'pointcloud.npz')
    input_ptcld_dict = np.load(pointcld_path, mmap_mode='r')
    input_pointcld = input_ptcld_dict['points'].astype(np.float32)
    input_normals = input_ptcld_dict['normals'].astype(np.float32)

    
    hvc_metadata_path = os.path.join(DATA_PATH, 'hard_vc_metadata.txt')
    hvc_meta = np.loadtxt(hvc_metadata_path)
    hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
    input_pointcld = apply_rotate(input_pointcld, hvc_rotate_dict)
    input_normals = apply_rotate(input_normals, hvc_rotate_dict)

    metadata_path = os.path.join(DATA_PATH, 'metadata.txt')
    meta = np.loadtxt(metadata_path)
    rotate_dict = {'elev': meta[10][1], 'azim': meta[10][0]-180}

    input_pointcld = apply_rotate(input_pointcld, rotate_dict)
    input_normals = apply_rotate(input_normals, rotate_dict)

    return input_pointcld, input_normals


def apply_rotate(input_points, rotate_dict):
    '''
    Azimuth rotation and elevation

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    theta_azim = rotate_dict['azim']
    theta_elev = rotate_dict['elev']
    theta_azim = np.pi+theta_azim/180*np.pi
    theta_elev = theta_elev/180*np.pi
    r_elev = np.array([[1,       0,          0],
                       [0, np.cos(theta_elev), -np.sin(theta_elev)],
                       [0, np.sin(theta_elev), np.cos(theta_elev)]])
    r_azim = np.array([[np.cos(theta_azim), 0, np.sin(theta_azim)],
                       [0,               1,       0],
                       [-np.sin(theta_azim),0, np.cos(theta_azim)]])
    rotated_points = r_elev@r_azim@input_points.T
    return rotated_points.T


def sample_points(input_points, input_vals):
    '''
    Samples a subset of points
    Args:
        input_points: 3D coordinates of points
        input_vals: corresponding occ/sdf values
    Returns:
        selected_points: 3D coordinates of subset of points 
        selected_vals: corresponding occ/sdf values of selected points

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    idx = np.arange(len(input_points))
    selected_points = input_points[idx, :]
    selected_vals = input_vals[idx]
    return selected_points, selected_vals


def get_sdf_h5(sdf_h5_file):
    '''
    This function reads sdf files saved in h5 format.
    
    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if ('pc_sdf_original' in h5_f.keys()
                and 'pc_sdf_sample' in h5_f.keys()
                and 'norm_params' in h5_f.keys()):
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            ori_pt = ori_sdf[:,:3]
            ori_sdf_val = None
            if sample_sdf.shape[1] == 4:
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
            else:
                sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]
        else:
            raise Exception('no sdf and sample')
    finally:
        h5_f.close()
    return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params



def eval_mesh(mesh, pointcloud_gt, points, val_gt, \
                n_points=300000, \
                sdf_val=None, iso=0.003):
    '''
    Borrowed and modified from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.

    Computes metric on point cloud
    Args:
        mesh: generated mesh
        pointcloud_gt: ground truth pointcloud (dimension Nx3)
        points: 3D coordinates of points (dimension Nx3)
        val_gt: ground truth occ/sdf
        n_points: number of points to sample from generated mesh
        sdf_val: predicted sdf values
        iso: isosurface value used in ground truth mesh generation
    Returns:
        metric dictionary contains:
            iou: regular point IoU for occ; regular point IoU with sign IoU \
                for sdf
            cd: Chamfer distance
            completeness: mesh completeness d(target->pred)
            accuracy: mesh accuracy d(pred->target)
            fscore: Fscore from 6 different thresholds
            precision: accuracy < threshold
            recall: completeness < threshold
    '''

    print("New PC contains ", len(mesh), "points.")
    if len(mesh) < n_points:
        replace=True
    else:
        replace = False
    idx = np.random.choice(len(mesh), n_points, replace=replace)
    pointcloud = mesh[idx]

    # Realign pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointcloud_gt)
    reg_p2p = o3d.registration.registration_icp(pcd, pcd_gt, 0.02)
    pcd = pcd.transform(reg_p2p.transformation)
    pointcloud = pcd.points

    # Eval pointcloud
    # Completeness: how far are the points of the target point cloud
    # from the predicted point cloud
    print("Evaluating Completeness.")
    completeness = distance_p2p(pointcloud_gt, pointcloud)

    # Accuracy: how far are the points of the predicted pointcloud
    # from the target pointcloud
    print("Evaluating Accuracy.")
    accuracy = distance_p2p(pointcloud, pointcloud_gt)

    # Get fscore
    print("Getting f-score.")
    fscore_array, precision_array, recall_array = [], [], []
    for thres in [0.5, 1, 2, 5, 10, 20]:
        fscore, precision, recall = calculate_fscore(\
            accuracy, completeness, thres/100.)
        fscore_array.append(fscore)
        precision_array.append(precision)
        recall_array.append(recall)
    fscore_array = np.array(fscore_array, dtype=np.float32)
    precision_array = np.array(precision_array, dtype=np.float32)
    recall_array = np.array(recall_array, dtype=np.float32)

    accuracy = accuracy.mean()

    completeness = completeness.mean()

    cd = completeness + accuracy

    iou = np.nan

    # sdf iou
    if sdf_val is not None:
        print("Computing SDF IoU.")
        sdf_iou, _, _ = compute_acc(sdf_val,val_gt)
    else:
        sdf_iou = np.nan
    iou = np.array([iou, sdf_iou])

    return {'iou': iou, 'cd': cd, 'completeness': completeness,\
                'accuracy': accuracy, \
                'fscore': fscore_array, 'precision': precision_array,\
                'recall': recall_array}


def distance_p2p(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Modified from 3DShapeGen

    Args:
        points_src: source points
        points_tgt: target points

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    kdtree = KDTree(points_tgt)
    dist, _ = kdtree.query(points_src)

    return dist


def calculate_fscore(accuracy, completeness, threshold):
    '''
    Calculate FScore given accuracy, completeness and threshold

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    recall = np.sum(completeness < threshold)/len(completeness)
    precision = np.sum(accuracy < threshold)/len(accuracy)
    if precision + recall > 0:
        fscore = 2*recall*precision/(recall+precision)
    else:
        fscore = 0
    return fscore, precision, recall


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    
    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1_temp = copy.deepcopy(occ1)
    occ2_temp = copy.deepcopy(occ2)
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    # Avoid dividing by 0
    if (area_union == 0).any():
        return 0.

    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)
    if isinstance(iou, (list, np.ndarray)):
        iou = np.mean(iou, axis=0)

    return iou


def compute_acc(sdf_pred, sdf, thres=0.01, iso=0.003):
    '''
    This function computes metric for sdf representation
    Args:
        sdf_pred: predicted sdf values
        sdf: gt sdf values
        thres: threshold to compute accuracy
        iso: iso value when generating ground truth meshes
    Returns:
        acc_sign: sign IoU where the signs of sdf_pred matches with sdf
        acc_thres: portion of points where sdf_pred is within thres from sdf
        iou: regular point IoU

    Borrowed from 3DShapeGen https://github.com/rehg-lab/3DShapeGen.
    '''
    sdf_pred = np.asarray(sdf_pred)
    sdf = np.asarray(sdf)

    acc_sign = (((sdf_pred-iso) * (sdf-iso)) > 0).mean(axis=-1)
    acc_sign = np.mean(acc_sign, axis=0)

    occ_pred = sdf_pred <= iso
    occ = sdf <= iso

    iou = compute_iou(occ_pred, occ)

    acc_thres = (np.abs(sdf_pred-sdf) <= thres).mean(axis=-1)
    acc_thres = np.mean(acc_thres, axis=0)
    return acc_sign, acc_thres, iou


def parameter_loss(projected_points, base_pc):
    chamfer_loss, _ = chamfer_distance(projected_points, base_pc)
    base_x = base_pc[0, :, 0].max() - base_pc[0, :, 0].min()
    base_y = base_pc[0, :, 1].max() - base_pc[0, :, 1].min()
    base_z = base_pc[0, :, 2].max() - base_pc[0, :, 2].min()
    loss = chamfer_loss 
    return loss, chamfer_loss


def point_cloud_from_depth(depth, fov, z_t):
    # Distance factor from the cameral focal angle
    factor = 2.0 * np.tan(fov / 2.0)

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # Valid depths are defined by the camera clipping planes
    #min_depth = np.unique(depth)[1]
    valid = (depth > 0) & (depth < 100)

    # Negate Z (the camera Z is at the opposite)
    z = -depth[valid]
    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows, cols)
    x = -(factor * (-depth) * (c - (cols / 2)) / ratio)[valid]
    y = (factor * (-depth) * (r - (rows / 2)) / ratio)[valid]
    z += z_t
    return np.dstack((x, y, z))


def point_cloud_from_depth_pt(depth, fov, z_t):
    # Distance factor from the cameral focal angle
    factor = (2.0 * torch.tan(fov / 2.0)).cuda()

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    c = torch.FloatTensor(c)
    r = torch.FloatTensor(r)
    # Valid depths are defined by the camera clipping planes
    #min_depth = torch.unique(depth)[1]
    valid = (depth > 0) & (depth < 100)

    # Negate Z (the camera Z is at the opposite)
    z = -depth[valid]
    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows, cols)

    x = -(factor * (-depth) * (c.cuda() - (cols / 2)) / ratio)[valid] 
    y = (factor * (-depth) * (r.cuda() - (rows / 2)) / ratio)[valid] 
    z += z_t
    return torch.dstack((x, y, z))
