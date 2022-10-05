"""
Use full_list.txt to create jobs for each workspace
Lists from:
    https://github.com/LPDI-EPFL/masif/tree/master/data/masif_site/lists
"""

import signac
import numpy as np
import pandas as pd
import glob
from Bio.PDB import PDBList
from tqdm import tqdm
from collections import defaultdict
import json
import shutil
import os


def download_pdb(pdb_id, pdir):
    pdb_list = PDBList(server='http://ftp.wwpdb.org', verbose=False)
    return pdb_list.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=pdir)


if __name__ == "__main__":
    pr = signac.get_project()

    path_full_list = 'source/DBD_lists/full_list.csv'
    pdb_list = pd.read_csv(path_full_list)
    pdb_list[['pdb_id', 'l', 'r']] = pdb_list.pdb_id.str.split('_', expand=True)

    path_key_list = 'source/DBD_lists/key.txt'
    keys = np.loadtxt(path_key_list, dtype=str,delimiter=',')

    # Some PDB IDs are outdated and need to be replaced
    obselete = ''

    for index, row in tqdm(pdb_list.iterrows(), total=pdb_list.shape[0]):
        sp_dict = dict(row)
        chain_ids = []
        chain_ids.append(sp_dict.pop('l'))
        chain_ids.append(sp_dict.pop('r'))
        sp_dict['chain_ids'] = chain_ids
        sp_dict['subset'] = ["DBD"]
        job = pr.open_job(sp_dict)
        if job.isfile('raw.pdb') is False:
            try:
                job.init()
                pdb_fname = download_pdb(job.sp.pdb_id, job.fn(''))
                shutil.move(job.fn(pdb_fname), job.fn("raw.pdb"))
            except FileNotFoundError:
                obselete += sp_dict['pdb_id'] + '_' + pdb_chain[0] + '_' + pdb_chain[1] + '\n'
        if os.path.isdir(job.fn('labels')) is False:
            os.mkdir(job.fn('labels'))

    with open('source/DBD_lists/obselete.txt', 'w') as f:
        f.write(obselete)

