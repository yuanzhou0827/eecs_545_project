conda create -n eecs545-pre
conda activate eecs545-pre
conda install -c conda-forge freud numpy signac signac-flow biopython h5py rowan pandas

python -m signac init preprocess
