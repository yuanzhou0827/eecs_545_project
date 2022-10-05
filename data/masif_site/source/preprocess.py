"""
Functions for data processing.
"""

def clean_pdb(job):
        from pdbfixer import PDBFixer
        from simtk.openmm.app import PDBFile
        fixer = PDBFixer(filename=job.fn("raw.pdb"))
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  #pH, default 7.0
        PDBFile.writeFile(
            fixer.topology, 
            fixer.positions, 
            open('fixed.pdb', 'w'),
            keepIds=True)


def extract(job):
    from .masif.extractPDB import extractPDB
    extractPDB(job.fn("fixed.pdb"), job.fn(f"protein_r.pdb"), job.sp.chain_ids[0])
    extractPDB(job.fn("fixed.pdb"), job.fn(f"protein_l.pdb"), job.sp.chain_ids[1])


def generate_msms(job, id=0, fname=None):
    from .masif.xyzrn import output_pdb_as_xyzrn
    from .masif.read_msms import read_msms

    import subprocess
    import tempfile
    import signac
    import h5py
    import numpy as np
    from .workflow import environment

    # Bio.PDB throws lots of warnings any time anything is malformed, but I'm
    # OK with it.
    import warnings
    warnings.filterwarnings("ignore")

    if fname is None:
        fname = f'protein_{id}.pdb'

    with tempfile.NamedTemporaryFile(prefix="tri", suffix=".xyzrn", dir=job.fn('')) as pdb_to_xyzr:
        # Generate temporary xyzrn file
        output_pdb_as_xyzrn(job.fn(fname), pdb_to_xyzr.name)
        subprocess.check_output(
            "{} -if {} -of {} -af {} -density {} -hdensity {}".format(
                environment.msms,
                pdb_to_xyzr.name,
                f'tri_{id}',
                f'tri_{id}',
                1.0,
                3.0
            ), shell=True)
        verts, faces, normals, names = read_msms(f"tri_{id}")
    return verts, faces, normals, names


def generate_surface(job):
    from .workflow import utils
    for id in ('l', 'r'):
        verts, faces, normals, names = generate_msms(job, id)
        utils.write(job, f'verts/{id}', verts)
        utils.write(job, f'faces/{id}', faces)
        utils.write(job, f'normals/{id}', normals)
        utils.write(job, f'names/{id}', names)


def contact_labels(job):
    import h5py
    import freud
    import numpy as np
    import signac
    from .workflow import utils
    pr = signac.get_project()
    verts, faces, _, _ = generate_msms(job, fname='fixed.pdb')

    for id in ('l', 'r'):

        with h5py.File("signac_data.h5", 'r') as f:
            verts_unit = f[f'verts/{id}'][:]

        is_interface = np.zeros_like(verts_unit).astype(int)

        max_dim = np.amax(np.abs(verts))

        # Use freud to find which points of chain r are within the specified threshold of subunit l
        box = freud.Box.cube(max_dim*5)  # Make box much larger than protein to avoid periodic boundaries

        # From MASIF, any point with distance**2 >= 2.0 from the regular mesh is an interface
        # This should be faster and more efficient than what MASIF uses since freud
        # does not need to compute the distances between every point to find the neighbor list
        nl = freud.AABBQuery(box, verts).query(verts_unit, {"r_max": np.sqrt(2)}).toNeighborList()
        is_interface[nl.neighbor_counts == 0] = 1
        utils.write(job, f'is_interface/{id}', is_interface)

