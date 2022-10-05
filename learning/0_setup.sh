git submodule update --init --recursive
module load cuda/10.1.243
conda create -n eecs545-train
conda activate eecs545-train
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install -c conda-forge numpy scikit-learn 
python -m pip install pyyaml
