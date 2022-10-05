import numpy as np
import signac
import h5py
import matplotlib.pyplot as plt
import argparse
import random

rng = random.Random(32)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--pdb_id', '-id',
    default='4FP8'
    )
parser.add_argument(
    '--n_sample', '-n',
    default=4096,
    type=int
    )
parser.add_argument(
    '--use_r', '-r',
    action="store_true"
    )

args = parser.parse_args()

pr = signac.get_project()

if args.use_r:
    unit = 'r'
else:
    unit = 'l'

# There should really only be one job associated with the id right now.
# For datasets with more variations (id, same pdb_id but with bound and 
# unbound), you will need to add more arguments the dictionary to 
# find those specific ones
for job in pr.find_jobs({"pdb_id": args.pdb_id}):
    print(job)

with h5py.File(job.fn('signac_data.h5'), 'r') as f:
    verts = f[f'verts/{unit}'][:]
    faces = f[f'faces/{unit}'][:]
    labels = f[f'is_interface/{unit}'][:, 0]

if len(labels) < args.n_sample:
    sample_idx = rng.choices(range(len(labels)), args.n_sample)
else:
    sample_idx = rng.sample(range(len(labels)), args.n_sample)

verts = verts[sample_idx]
labels = labels[sample_idx]

x = verts[:, 0]
y = verts[:, 1]
z = verts[:, 2]

    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=labels, cmap='cividis', alpha=1.0)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
print(args.pdb_id)
plt.show()
