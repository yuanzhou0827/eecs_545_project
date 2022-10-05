import numpy as np
import h5py
import warnings
import os
from torch.utils.data import Dataset
import torch
warnings.filterwarnings('ignore')


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # point = point[centroids.astype(np.int32)]
    return centroids.astype(np.int32)


class MasifSiteDataset(Dataset):
    def __init__(
            self,
            root,
            rng,
            dset='DBD',
            npoint=1024,
            split='train',
            uniform=False,
            normal_channel=True):
        self.root = root
        self.rng = rng
        self.npoints = npoint
        self.uniform = uniform
        self.split = split

        self.cat = ['non_interface', 'interface']
        self.classes = {
            'non_interface': 0,
            'interface': 1
        }
        self.normal_channel = normal_channel

        assert (split == 'train' or split == 'test')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = os.path.join(self.root, f'{dset}.h5')

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        with h5py.File(self.datapath, 'r') as f:
            point_set = f[f"{self.split}/point_set/{index}"][:].astype(np.float32)
            labels = f[f"{self.split}/labels/{index}"][:, 0].astype(np.int32)

        if self.uniform:
            subsample_idx = farthest_point_sample(point_set[:, :3], self.npoints)
        else:
            n_point_size = len(point_set)
            subsample_idx = list(range(n_point_size))
            if n_point_size >= self.npoints:
                subsample_idx = self.rng.sample(subsample_idx, self.npoints)
            else:
                subsample_idx = self.rng.choices(subsample_idx, k=self.npoints)
            subsample_idx = np.array(subsample_idx)

        point_set = point_set[subsample_idx]
        labels = labels[subsample_idx]
        point_set = torch.Tensor(point_set)
        labels = torch.Tensor(labels)

        return point_set.cuda(), labels.cuda()

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = MasifSiteDataset('../../data/', split='train', uniform=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, drop_last=True)
    for point, labels in DataLoader:
        import ipdb; ipdb.set_trace()
        print(point.shape)
        print(labels.shape)

