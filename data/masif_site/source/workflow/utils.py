from contextlib import redirect_stderr, redirect_stdout, contextmanager
import signac.contrib.job
from typing import Tuple
import h5py


@contextmanager
def redirect_with_job(job, op_name, redirect_output=True, mode='w'):
    """Context manager that enters a job workspace and redirects output into
    files for the specified operation."""
    @contextmanager
    def file_redirection():
        if redirect_output:
            with open(job.fn('stderr_{}.txt'.format(op_name)), mode) as err:
                with open(job.fn('stdout_{}.txt'.format(op_name)),
                          mode) as out:
                    with redirect_stderr(err):
                        with redirect_stdout(out):
                            yield
        else:
            yield

    with file_redirection():
        with job:
            yield


def write(
        job: signac.contrib.job.Job,
        label: str,
        value):
    """Creates a dataset for a label if the label does not exist.
    """
    try:
        with h5py.File(job.fn('signac_data.h5'), mode='a') as f:
            f[label] = value
    except OSError:  # if label already exists
        with h5py.File(job.fn('signac_data.h5'), mode='a') as f:
            f[label][:] = value
