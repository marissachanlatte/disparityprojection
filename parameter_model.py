import torch
import torch.nn as nn
import numpy as np

import utils

class ParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # random initialization
        random_values = np.random.rand(4)
        random_values[1] += 2
        weights = torch.Tensor(random_values)
        # make weights torch parameters
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, depth, segmentation):
        ''' Function to be optimized (3D projection)'''
        fov, z_t, s, d = self.weights
        depth = s*(depth) + d
        depth = 1/depth
        depth[segmentation == 0] = 100
        return utils.point_cloud_from_depth_pt(depth, fov, z_t)
