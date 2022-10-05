"""
Utility function for PointConv
Originally from : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/utils.py
Modify by Wenxuan Wu
Date: September 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
import freud

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(
    npoint, nsample, xyz, points, 
    density_scale=None,
    ):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    D = points.shape[-1]
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdim = True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, D)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def grouping(feature, nsample, xyz, q_xyz, use_xyz=True):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    if q_xyz is None:
        q_xyz = xyz
    Bq, Nq, Cq = q_xyz.shape
    device = xyz.device

    points_in = xyz.cpu().numpy()
    query_points_in = q_xyz.cpu().numpy()
    points_in = points_in.reshape(-1, C)
    query_points_in = points_in.reshape(-1, Cq)
    max1 = np.amax(np.abs(points_in))
    max2 = np.amax(np.abs(query_points_in))
    max_dim = np.amax([max1, max2])
    box = freud.Box.cube(max_dim * 5)
    query_dict = {'mode': 'nearest', 'num_neighbors': nsample, 'exclude_ii': True}
    nl = freud.AABBQuery(box, points_in).query(
        query_points_in,
        query_dict
    ).toNeighborList()

    grouped_xyz = points_in[nl[:, 0]]
    grouped_query = query_points_in[nl[:, 1]]
    grouped_xyz -= grouped_query
    grouped_xyz = grouped_xyz.reshape(B, N, nsample, C)

    grouped_xyz = torch.Tensor(grouped_xyz).to(device).long()

    #grouped_xyz = index_points(xyz, s_idx) # [B, npoint, nsample, C]
    #grouped_xyz = grouped_xyz - index_points(q_xyz, idx)
    idx = torch.Tensor(nl[:, 1].astype(int) % Nq).to(device).long()
    idx = idx.view(B, N, nsample)

    grouped_feature = index_points(feature, idx)
    if use_xyz:
        Bf, Nf, Cf = feature.shape
        new_points = torch.cat(
                [grouped_xyz, grouped_feature], 
                dim=-1
            ) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_feature


    return grouped_xyz, new_points, idx

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density


def three_nn(query_points, points):
    device = points.device
    B, N, C = points.shape
    Bq, Nq, D = query_points.shape

    points_in = points.cpu().view(-1, C).numpy()
    query_points_in = query_points.cpu().view(-1, D).numpy()

    max1 = torch.max(torch.abs(points))
    max2 = torch.max(torch.abs(query_points))
    max_dim = torch.maximum(max1, max2).cpu().numpy()
    box = freud.Box.cube(max_dim * 5)
    query_dict = {'mode': 'nearest', 'num_neighbors': 3, 'exclude_ii': True}
    nl = freud.AABBQuery(box, points_in).query(
        query_points_in,
        query_dict
        ).toNeighborList()

    sq_dist = torch.Tensor(nl.distances ** 2).to(device).long()
    idx = torch.Tensor(nl[:, 1].astype(int) % Nq).to(device).long()
    sq_dist, idx = sq_dist.view(Bq, Nq, D), idx.view(Bq, Nq, D)
    return sq_dist, idx


class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)
        
        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

class PointConvSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all):
        super(PointConvSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz_norm, _ = sample_and_group(self.npoint, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density 

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points, inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, inverse_density.view(B, N, 1))
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        inverse_max_density = grouped_density.max(dim = 2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)     
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointConvFP(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bandwidth, npoints=4096,
        last_channel=None
    ):
        super(PointConvFP, self).__init__()
        self.npoints = npoints
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if last_channel is None:
            last_channel = mlp[0] 
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 32)
        #self.weightnet = WeightNet(3, in_channel)
        self.np_conv = nn.Conv2d(in_channel, mlp[0], 1, stride=(1, 512  + 3))
        #self.linear = nn.Linear(mlp[-1] * (64 + 3), mlp[-1])
        #self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet([16, 1])
        self.bandwidth = bandwidth

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, _, N = xyz1.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        #dist, idx = three_nn(xyz1, xyz2)
        dist, idx = three_nn(xyz2, xyz1)
        mask = dist < 1e10
        dist = dist[mask].view(mask.shape)
        norm = torch.sum((1.0/dist), 2, keepdim=True)
        weight = (1.0 / dist) / norm
        i, j, k = weight.shape
        if N == self.npoints:
            points2 = points2.permute(0, 2, 1)
        interpolated_points = index_points(points2, idx) * weight.view(i, j, k, 1)
        interpolated_points = torch.sum(interpolated_points, dim=2)

        grouped_xyz, grouped_xyz_norm, idx = grouping(interpolated_points, self.nsample, xyz1, xyz2)

        xyz_density = compute_density(xyz1, self.bandwidth)
        inverse_density = 1.0 / xyz_density
        grouped_density = index_points(inverse_density.view(B, N, 1), idx)
        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density

        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1).float()
        weight = self.weightnet(grouped_xyz)
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))

        new_points = grouped_xyz * density_scale
        new_points = torch.matmul(
            new_points.permute(0, 3, 1, 2),
            weight.permute(0, 3, 2, 1)
        )
        if N == self.npoints:
            new_points = new_points.permute(0, 2, 1, 3)
        new_points = torch.squeeze(self.np_conv(new_points))
        if N == self.npoints:
            points1 = points1.permute(0, 2, 1)
            new_points = new_points.permute(0, 2, 1)

        if points1 is not None:
            new_points1 = torch.cat([new_points, points1], dim=-1)
            i, j, k = new_points1.shape
            new_points1 = new_points1.view(i, j, k, 1)
        else:
            new_points1 = new_points


        if N == self.npoints:
            new_points1 = new_points1.permute(0, 2, 1, 3)
        # new_xyz: sampled points position data, [B, npoint, C]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points1 = F.relu(bn(conv(new_points1)))
        """
        new_points1 = self.linear(new_points1.view(B, -1))
        new_points1 = self.bn_linear(new_points1)
        new_points1 = F.relu(new_points1)
        """

        return torch.squeeze(new_points1)
