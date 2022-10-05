"""
Takes raw pdb files and prepares them for the model.
TODO: Make precondition/postcondition labels for brevity
"""
from source.workflow import environment, utils, labels
from source import preprocess
import flow
import os

class Project(labels.Project):
    pass


setup = Project.make_group("setup")


@setup
@Project.operation
@Project.pre.isfile("raw.pdb")
@Project.post(labels.pdb_fixed)
@flow.with_job
def pdb_fixer(job):
    """
    pdb fixer to protonate and replace missing atoms
    """
    labels.track_progress(preprocess.clean_pdb, job, 'pdb_fixer')


@setup
@Project.operation
@Project.pre(labels.pdb_fixed)
@Project.post(labels.chains_extracted)
@flow.with_job
def extract_chains_from_pdb(job):
    """
    "Extract selected chains from a PDB and save the extracted chains to an output file."
    From https://github.com/LPDI-EPFL/masif/
    """
    labels.track_progress(preprocess.extract, job, 'extract_chains_from_pdb')


@setup
@Project.operation
@Project.pre(labels.chains_extracted)
@Project.post(labels.generated_surface)
@flow.with_job
def generate_surface(job):
    """
    Read a pdb file and output it is in xyzrn for use in MSMS,
    then generate surface representation of protein using MSMS.
    From https://github.com/LPDI-EPFL/masif/
    """

    labels.track_progress(preprocess.generate_surface, job, 'generate_surface')



@setup
@Project.operation
@Project.pre(labels.generated_surface)
@Project.post(labels.contacts_labeled)
@flow.with_job
def contact_labels(job):
    """Label each vertex as an interface or non-interface point.
    """
    labels.track_progress(preprocess.contact_labels, job, 'contact_labels')


@setup
@Project.operation
@Project.pre(labels.generated_surface)
@Project.post(labels.point_cloud_centered)
@flow.with_job
def center_point_cloud(job):
    import h5py
    import freud
    import numpy as np
    try:
        for id in ['l', 'r']:
            with h5py.File('signac_data.h5', 'r') as f:
                verts = f[f'verts/{id}'][:]
            # Make box way larger than point cloud to avoid periodic BC
            box = freud.Box.cube(np.amax(np.abs(verts))*5)
            # center of mass = centroid for point cloud
            centered_verts = box.center(verts)
            utils.write(job, f'centered_verts/{id}', centered_verts)
        with open('labels/center_point_cloud_success.txt', 'w') as _:
            pass
    except:
        with open('labels/center_point_cloud_fail.txt', 'w') as _:
            pass


@setup 
@Project.operation
@Project.pre(labels.chains_extracted)
@flow.with_job
def set_up_rotations(job):
    import rowan
    quat = rowan.random.rand(10)
    utils.write(job, 'rotations', rowan.normalize(quat))


if __name__ == '__main__':
    Project().main()
