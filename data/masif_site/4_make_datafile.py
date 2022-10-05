import signac
import h5py
import numpy as np
import rowan
import random
import os.path

from tqdm import tqdm

path = "source/DBD_lists/"
pr = signac.get_project()
rng = random.Random(2048)

if os.path.isfile('../DBD.h5'):
    os.rm('../DBD.h5')

for split in ('train', 'test'):
    fpath = path + f"{split}_list.txt"
    split_list = np.genfromtxt(fpath, delimiter=',', dtype=str)
    n_samples_w_row = list(range(len(split_list) * 10))
    rng.shuffle(n_samples_w_row)

    print(split)
    for n in tqdm(n_samples_w_row):
        job_id, unit = split_list[n % len(split_list)]
        job = pr.open_job(id=job_id)
        with h5py.File(job.fn("signac_data.h5"), 'r') as f:
            points = f[f'centered_verts/{unit}'][:]
            labels = f[f'is_interface/{unit}'][:]
            quat = f[f'rotations'][n // len(split_list)]
        quat = rowan.normalize(quat)
        points = rowan.rotate(quat, points)
        with h5py.File('../DBD.h5', 'a') as f:
            f[f'{split}/point_set/{n}'] = points
            f[f'{split}/labels/{n}'] = labels

    


