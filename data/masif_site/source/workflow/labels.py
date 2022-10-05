from flow import FlowProject
import h5py
import os
import traceback
"""
Labels for data preprocessing operations.
"""


"""Hook system to track operation failures/success"""


def track_progress(foo, job, operation_name):
    try:
        foo(job)
        with open(job.fn(f"labels/{operation_name}_success.txt"), 'w') as _:
            pass
    except Exception:
        with open(job.fn(f"labels/{operation_name}_fail.txt"), 'w') as f:
            traceback.print_exc(file=f)


class Project(FlowProject):
    pass


@Project.label
def pdb_fixed(job):
    operation_name = "pdb_fixer"
    key = "success"
    check_hook_file = job.isfile(f"labels/{operation_name}_{key}.txt")
    return check_hook_file


@Project.label
def pdb_fixed_error(job):
    operation_name = "pdb_fixer"
    key = "fail"
    check_hook_file = job.isfile(f"labels/{operation_name}_{key}.txt")
    return check_hook_file


@Project.label
def chains_extracted(job):
    operation_name = "extract_chains_from_pdb"
    key = "success"
    check_hook_file = job.isfile(f"labels/{operation_name}_{key}.txt")
    return check_hook_file


@Project.label
def chains_extracted_failed(job):
    operation_name = "extract_chains_from_pdb"
    key = "fail"
    check_hook_file = job.isfile(f"labels/{operation_name}_{key}.txt")
    return check_hook_file


@Project.label
def generated_surface(job):
    operation_name = "generate_surface"
    key = "success"
    check_hook_file = job.isfile(f"labels/{operation_name}_{key}.txt")
    return check_hook_file


@Project.label
def generated_surface_fail(job):
    operation_name = "generate_surface"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def computed_charges(job):
    # TODO: update with outputes
    operation_name = "compute_charges"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def computed_charges_failed(job):
    operation_name = "compute_charges"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def pqr_generated(job):
    # TODO: update with outputes
    operation_name = "gen_pqr"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def pqr_generated_failed(job):
    operation_name = "gen_pqr"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def apbs_computed(job):
    # TODO: update with outputes
    operation_name = "apbs_electrostatics"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def apbs_computed_failed(job):
    operation_name = "apbs_electrostatics"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def charges_extracted(job):
    # TODO: update with outputes
    operation_name = "charges_extracted"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def charged_extracted_failed(job):
    operation_name = "charges_extracted"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def contacts_labeled(job):
    # TODO: update with outputes
    operation_name = "contact_labels"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def contacts_labeled_failed(job):
    operation_name = "contact_labels"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def point_cloud_centered(job):
    operation_name = "center_point_cloud"
    key = "success"
    return job.isfile(f"labels/{operation_name}_{key}.txt")


@Project.label
def point_cloud_centered_failed(job):
    operation_name = "center_point_cloud"
    key = "fail"
    return job.isfile(f"labels/{operation_name}_{key}.txt")
