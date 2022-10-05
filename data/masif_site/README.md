To setup environment:

conda create -n eecs545-pre
conda activate eecs545-pre
conda install -c conda-forge freud numpy signac signac-flow biopython h5py rowan pandas pdbfixer


Initialize project: 

python -m signac init preprocess

Or ``0_create_environment.sh`` to setup the environment and initialize the project:

. 0_create_environment.sh


Install MSMS and update path the msms in source/workflow/environments.py
Available here: https://ccsb.scripps.edu/mgltools/downloads/


Install PDBFixer. 
Instructions here:
https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
Note that one of PDBFixer's depenencies is OpenMM. This package should be installable through conda. Unfortunately, on GreatLakes, OpenMM needs to be built.


To check which steps need to be fulfilled in data preprocessing:
python 2_project.py status

To submit job scripts, see:
python 2_project.py submit --help
