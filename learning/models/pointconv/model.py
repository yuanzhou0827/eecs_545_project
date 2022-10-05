
  
"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils.pointconv_util import PointConvDensitySetAbstraction,PointConvFP

class get_model(nn.Module):
    def __init__(self, num_classes=1, npoints=4096, normal_channel = False):
        super(get_model, self).__init__()
        self.npoints = npoints
        if normal_channel:
            feature_dim = 6
        else:
            feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=1024,
            nsample=32,
            in_channel=feature_dim + 3,
            mlp=[32, 32, 64],
            bandwidth=0.1,
            group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=256,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            bandwidth=0.2,
            group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=64,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            bandwidth=0.4,
            group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(
            npoint=36,
            nsample=32,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
            bandwidth=0.4,
            group_all=False)

        self.fp1 = PointConvFP(
            nsample=16,
            in_channel=64,
            mlp=[256, 512],
            bandwidth=0.8,
        )
        self.fp2 = PointConvFP(
            nsample=16,
            in_channel=256,
            mlp=[128, 256, 256],
            bandwidth=0.4,
        )
        self.fp3 = PointConvFP(
            nsample=16,
            in_channel=1024,
            mlp=[64, 256, 256, 128],
            bandwidth=0.2,
        )
        self.fp4 = PointConvFP(
            nsample=16,
            in_channel=3,
            last_channel=128 + 3,
            mlp=[128, 128, 128],
            bandwidth=0.1,
        )
        self.fc1 = nn.Linear(128, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)
        #l1_points = l1_points.permute(0, 2, 1)
        points = self.fp4(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.drop1(
            F.relu(
                self.bn1(
                    self.fc1(points.permute(0, 2, 1)).permute(0, 2, 1)
                    )
                )
            )
        x = self.drop2(
                F.relu(
                    self.bn2(self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1))
                )
            )
        x = self.fc3(x.permute(0, 2, 1))
        x = F.log_softmax(x, -1)
        return x.view(B, self.npoints, 1), None

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = nn.BCELoss()(pred, target)
        return total_loss
