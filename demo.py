import os
import copy

import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

import utils
from parameter_model import ParameterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_ply(name, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(name, pcd)

def training_loop(depth, segmentation, base_pc):
    model = ParameterModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    epoch_it = 0
    weights = torch.Tensor([0, 0, 0, 0])
    loss = 1000
    # Training loop
    while True:
        epoch_it += 1
        model.train()
        opt.zero_grad()
        projected_points = model(depth, segmentation)
        loss, chamfer_loss = utils.parameter_loss(projected_points, base_pc)
        print("Loss, ", loss)
        loss.backward()
        opt.step()

        # f, s, d, z_t
        old_weights = copy.deepcopy(weights.detach())
        weights = copy.deepcopy(model.weights)
        if torch.abs(weights - old_weights).max() < 0.0001:
           break
    return weights, loss, chamfer_loss


def vis_occ_split(pointcloud):
    # Split into visible & occluded
    diameter = np.linalg.norm(np.asarray(pointcloud.get_max_bound()) - np.asarray(pointcloud.get_min_bound()))
    camera = [0, 0, 3]
    radius = diameter * 1e4
    occ, vis = pointcloud.hidden_point_removal(camera, radius)

    points = np.asarray(pointcloud.points)
    vis_pcd = points[vis]
    occ_pcd = np.delete(points, vis, axis=0)
    return vis_pcd, occ_pcd


def remove_points(new_visible, init_occ):
    # remove no longer occluded points
    old_pointcloud = np.array([0])
    new_pointcloud = np.concatenate((new_visible, init_occ))
    while (old_pointcloud.shape[0] != new_pointcloud.shape[0]):
        old_pointcloud = new_pointcloud
        _, new_occ = utils.occ_vis_split(old_pointcloud) 
        new_pointcloud = np.concatenate((new_visible, new_occ))
        new_pointcloud = np.unique(new_pointcloud, axis=0)
    return new_pointcloud


def evaluate_pointcloud(geometry, metrics_path):
    # Get SDF GT
    points, val_gt = utils.get_points_sdf_sample()
    # Get Point Cloud GT
    pointcloud_gt, _ = utils.get_pointcloud_sample()
    # Eval 
    out_dict = utils.eval_mesh(geometry, pointcloud_gt, 
                                        points, val_gt)
    # Save
    np.savez(metrics_path, img_name='demo', cd=out_dict['cd'], \
                 iou=out_dict['iou'], fscore=out_dict['fscore'])


def main():
    out_dir = 'demo_out'
    data_prefix = 'data'

    # Make Output Directory
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    img_path = os.path.join(data_prefix, 'input_img.png')

    # Copy Image for Reference
    os.system("cp %s %s" % (img_path, out_dir))

    # Load depth & convert to disparity
    seg_path = os.path.join(data_prefix, 'segmentation.png')
    depth_path = os.path.join(data_prefix, 'depth.exr')
    depth = cv2.imread(depth_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 1]
    segmentation = (depth > 0.1) & (depth < 100)
    depth[segmentation == 0] = 100
    depth = 1/depth
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    depth = torch.FloatTensor(depth).to(device)

    # Load base
    init_pred = o3d.io.read_triangle_mesh(os.path.join(data_prefix, 'initial_prediction.obj'))
    init_pcd = init_pred.sample_points_uniformly(number_of_points=50000)
    init_vis, init_occ = vis_occ_split(init_pcd)
    base_pc = torch.FloatTensor(init_vis).to(device)
    base_pc = torch.unsqueeze(base_pc, dim=0)

    # Fit to initial prediction
    best_loss = 100
    best_weights = 0
    all_losses = []
    for i in range(6):
        weights, loss, chamfer_loss = training_loop(depth, segmentation, base_pc)
        all_losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_chamfer = chamfer_loss
            best_weights = weights
    print("All losses: ", all_losses)
    print("Best Loss: ", best_loss)
    losses = np.array([loss, chamfer_loss])
    loss_path = os.path.join(out_dir, "loss.npy")
    np.save(loss_path, losses)

    # project new visible points
    fov, z_t, s, d = best_weights.detach().numpy()

    depth = depth.detach().cpu().numpy()
    depth = s*(depth) + d
    depth = 1/depth
    depth[segmentation == 0] = 100
    new_visible = utils.point_cloud_from_depth(depth, fov, z_t)[0]
    save_ply("first_proj.ply", new_visible)

    # Remove no longer occluded points
    new_pointcloud = remove_points(new_visible, init_occ)

    # Save as .ply 
    save_ply(os.path.join(out_dir,'finetuned.ply'), new_pointcloud)
    print("Saving ", os.path.join(out_dir,'finetuned.ply'))

    # Evaluate
    print("Evaluating Point Cloud.")
    metrics_path = os.path.join(out_dir, 'eval.npz')
    evaluate_pointcloud(new_pointcloud, metrics_path)

    print("Finished demo. Output saved to ", out_dir)
    return 0

if __name__ == '__main__':
    main()