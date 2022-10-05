import flow
"""
This contains paths and code to setup software and environments for data preprocessing.
"""

# Specify paths to environment and software here
pdb2xyzrn = "$HOME/local/msms/pdb_to_xyzrn"
msms = "$HOME/local/msms/msms.i86Linux2.2.6.1 "


def with_venv(func):
    """Creates decorator for flow functions.
    Activates the virtual environment "proteins_andes".
    """
    environment = flow.environment.get_environment()
    return flow.directives('conda activate eecs545-pre; python3')
