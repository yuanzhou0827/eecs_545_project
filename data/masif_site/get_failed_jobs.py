import argparse
import signac
from tqdm import tqdm

JOB_OPERATIONS = [
    "pdb_fixer",
    "extract_chains_from_pdb",
    "generate_surface",
    "compute_charges",
    "gen_pqr",
    "aps_electrostatics",
    "charges_extracted",
    "contact_labels",
    "center_point_cloud"
]


def parse_jobs(n, operation):
    pr = signac.get_project()

    if n is None:
        n = len(pr)

    failed_jobid = []
    for job in tqdm(pr):
        if job.isfile(f"labels/{operation}_fail.txt"):
            failed_jobid.append(job.id)
            if len(failed_jobid) > n:
                break
    print(failed_jobid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for job_operation in JOB_OPERATIONS:
        parser.add_argument(
            f'--{job_operation}',
            help=f"Get job ids for {job_operation}.",
            action="store_true"
        )
    parser.add_argument(
        '--n', '-n',
        help="Number of failed job ids to return. Default is all.",
        type=int,
        default=None
        )
    args = vars(parser.parse_args())

    n = args.pop('n')

    get_operations = []
    for job_operation in args:
        if args[job_operation]:
            get_operations.append(job_operation)

    for job_operation in get_operations:
        parse_jobs(n, job_operation)

