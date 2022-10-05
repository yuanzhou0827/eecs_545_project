import signac
import pandas as pd
import numpy as np
import h5py
import random
from tqdm import tqdm


rng = random.Random(32)
pr = signac.get_project()


def split(df, idx):

    n = df.shape[0]
    new_idx = np.array([[i % n, i // n] for i in idx])

    new_df = df.iloc[new_idx[:, 0]]

    ids = np.array(['l' for _ in range(len(new_idx))])

    ids[new_idx[:, 1] == 1] = 'r'

    outputs = []
    n = 0
    for index, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
        sp_dict = dict(row)
        chain_ids = []
        chain_ids.append(sp_dict.pop('l'))
        chain_ids.append(sp_dict.pop('r'))
        sp_dict['chain_ids'] = chain_ids
        sp_dict['subset'] = ['DBD']

        job = pr.open_job(sp_dict)
        assert job in pr

        add = False
        if job.isfile('labels/center_point_cloud_success.txt'):
            if job.isfile('labels/contact_labels_success.txt'):
                add = True

        if add:
            outputs.append([job.id, ids[n]])
        n += 1

    return outputs


def write_to_file(split_list, fname):
    f_output = ''
    rng.shuffle(split_list)
    for jobid, unit in split_list:
        f_output += f'{jobid},{unit}\n'
    with open(fname, 'w') as f:
        f.write(f_output)


path_full_list = 'source/DBD_lists/full_list.csv'
pdb_list = pd.read_csv(path_full_list)
pdb_list[['pdb_id', 'l', 'r']] = pdb_list.pdb_id.str.split('_', expand=True)

# Small dataset, use 80/20 split
n_total_samples = pdb_list.shape[0] * 2

train_list = []
test_list = []
for n in range(3):
    subset = pdb_list[pdb_list['difficulty'] == n]
    n_samples = subset.shape[0] * 2
    n_train = round(0.8 * n_samples)
    n_test = n_train - n_train

    idx = list(range(n_samples))
    rng.shuffle(idx)

    raw_train_idx = idx[:n_train]
    raw_test_idx = idx[n_train:]

    print(f"-- Train split for difficulty {n}")
    train_list.extend(split(subset, raw_train_idx))
    print(f"-- Test split for difficulty {n}")
    test_list.extend(split(subset, raw_test_idx))

print(f"Train list size: {len(train_list)}, Test list size: {len(test_list)}")
write_to_file(test_list, 'source/DBD_lists/test_list.txt')
write_to_file(train_list, 'source/DBD_lists/train_list.txt')


