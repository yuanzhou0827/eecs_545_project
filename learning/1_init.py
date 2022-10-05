import signac
import shutil

pr = signac.get_project()

all_models = ('pointnet', 'pointnetpp_msg', 'pointnetpp_ssg', 'pointconv')
batch_size = (16)
npoints = (4096)

default_sp_dict = {
    'dset': 'DBD',
    'npoint': 4096,
    'uniform': True,
    'batch_size': 16,
    'model': 'pointnet',
    'optimizer': 'SDG',
    'learning_rate': 0.001,
    'epoch': 25,
    'lr_decay': 0.5,
    'step_size': 20,
}


for model in all_models:
    sp_dict = default_sp_dict
    sp_dict['model'] = model
    job = pr.open_job(sp_dict)
    job.init()
    shutil.copy('../data/DBD.h5', job.fn('DBD.h5'))

sp_dict = default_sp_dict
sp_dict['uniform'] = False
job = pr.open_job(sp_dict)
job.init()
shutil.copy('../data/DBD.h5', job.fn('DBD.h5'))


