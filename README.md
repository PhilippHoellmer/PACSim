# Ionic Colloids

Install and setup [Git Large File Storage](https://git-lfs.com) (used for the large files in the Literature directory.)

## Dependencies

If you only want to use the openmm part of this package, you can use Python 3.12 (or any older version >= 3.10).

If you want to use the Hoomd or openmm-plumed parts of this package, use Python 3.10 because that is the latest Python version that is supported by Hoomd < 3 and openmm-plumed v2.

We recommend installing Python and the required packages using 
[Anaconda](https://www.anaconda.com/products/distribution). 

The following packages are required:

- jupyterlab >= 4.1 
- matplotlib >= 3.8
- numpy >= 1.26
- openmm >= 8.1
- pytest >= 7.4
- gsd >= 3.2
- pyyaml >= 6.0
- tqdm >= 4.65
- pandas >= 2.2
- hoomd == 2.9.7
- plumed == 2.9.0
- openmm-plumed v2
- scipy >= 1.16.1
- pymatgen >= 2024.10.3

Note that some PLUMED modules that are necessary for functionality such as calculation of local order parameters and adjacency matrices are not compiled by default, but can be enabled during configuration. 
We recommend enabling the crystallization, multicolvar, and adjmat modules when compiling:

```bash
./configure --enable-modules=crystallization+multicolvar+adjmat
```
 
PLUMED 2.9.0 is available for download [here](https://github.com/plumed/plumed2/releases/tag/v2.9.0). Instructions for configuring and compiling can be found [here](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html). 

After installing and configuring PLUMED, openmm-plumed can be installed by following [these instructions](https://github.com/openmm/openmm-plumed). To ensure that openmm-plumed points to the right version of PLUMED, we strongly recommend building the openmm-plumed plugin from scratch using CMake rather than doing a conda install.

If you are using PLUMED, it is not necessary to install the MPI version. However, one should carefully check that OpenMP
is enabled during the configuration step of PLUMED. Then, increasing `PLUMED_NUM_THREADS` should speed up the 
calculations and increase CPU usage.

For the CPU platform of OpenMM, it might make sense for small systems to set `OPENMM_CPU_THREADS=1`.

## Installation

Clone the repository and install the package in editable mode in your virtual environment using pip:

```bash
pip install -e .
```

Note that this attempts to install the requirements with pip, if you did not install them yourself before. However, because hoomd and openmm-plumed are not available on PyPI, you need to install them manually (or via conda). If plumed was not previously configured, it will be installed at this point with default modules and some functionality may not be available.

## Testing

After installation, you can test whether your installation is working correctly by running the following command from 
this directory:

```bash
pytest colloids
```

If hoomd is not installed, some tests are automatically skipped.

## Usage

The installation process creates three executables `colloids-run`, `colloids-resume`, and `colloids-create`. You might 
have to add the directory where pip installs executables to your PATH environment variable in order to access these 
executables.

### colloids-run

The `colloids-run` executable is used to run simulations. It expects a configuration file in yaml format as the only 
positional argument:

```bash
colloids-run run.yaml
```

An exemplary configuration file called `example.yaml` can be created with the command 
`colloids-run --example`. Another exemplary configuration file is provided in [`colloids/tests/run_test.yaml`](colloids/tests/run_test.yaml).

### colloids-resume
A simulation that is run with the `colloids-run` executable creates checkpoints in periodic intervals. One can resume a 
simulation from a checkpoint using the `colloids-resume` executable. It expects the original configuration file (because
the checkpoint file only stores the positions and velocities of the particles in an OpenMM context), the checkpoint 
file, and the number of time steps that should be run (the corresponding value in the configuration file is ignored). 
For example, use the following command to continue a simulation for 100000 time steps:

```bash
colloids-resume run.yaml checkpoint.chk 100000
```

### colloids-create
The configuration file for the `colloids-run` executable specifies the filename of an initial configuration for the 
simulation in the `initial_configuration` key. This initial configuration should be stored in the [GSD/HOOMD file 
format](https://www.ovito.org/docs/current/reference/file_formats/input/gsd.html#file-formats-input-gsd).

The `colloids-create` executable can be used to create an initial configuration for simulations in the GSD file 
format. It expects two positional arguments:
1. A configuration file that specifies the parameters of the initial configuration. See 
   [`colloids/colloids_create/configuration.yaml`](colloids/colloids_create/configuration.yaml) for an example. Another 
   exemplary configuration file called `example_configuration.yaml` can be created with the command
   `colloids-create --example`.
2. The name of the GSD file of the initial configuration. See  [`colloids/colloids_create/tests/reference_configuration.gsd`](colloids/colloids_create/tests/reference_configuration.gsd) for an example.

In the configuration yaml file, the `cluster_specification` key requires the filename of a LAMMPS data file specifying how
to construct the initial configuration. See [`colloids/colloids_create/cluster.lmp`](colloids/colloids_create/cluster.lmp) for an example.

A typical workflow for running a simulation with `colloids-run` from an initial configuration created by 
`colloids-create` consists of creating a directory with `run.yaml`, `configuration.yaml`, and 'cluster.lmp` files, 
and then running the following two commands:

```bash
colloids-create configuration.yaml first_frame.gsd
colloids-run run.yaml
```

### colloids-analyze

The `colloids-run` executable generates a trajectory in the GSD file format that can be visualized with 
[Ovito](https://www.ovito.org) and 
analyzed with the [GSD](https://gsd.readthedocs.io/en/stable/python-api.html) Python package.

In addition, the `colloids-run` executable generates a CSV file that contains the time series of the potential energy,
the kinetic energy, and the temperature of the system. The `colloids-analyze` executable can be used to plot these time
series. Here, it can plot the results of several simulations at once.

The `colloids-analyze` expects a configuration file in yaml format that specifies the parameters of the analysis (like 
the output directory where the plots should be generated) as the first positional argument. An exemplary configuration 
file called `example_analysis.yaml` can be created with the command `colloids-analyze --example`. Another exemplary 
configuration file is provided in [`colloids/colloids_analyze/analysis.yaml`](colloids/colloids_analyze/analysis.yaml).

After this, the `colloids-analyze` executable receives an arbitrary number of configuration files that specified the 
parameters of the simulations that should be analyzed. These configuration files contain the name of the CSV files
that will be plotted.

Assume, for example, that you ran three simulations with `colloids-run` in the directories `Run1`, `Run2`, and `Run3` 
based on configuration files called `run.yaml` in either of these directories. You can analyze and compare the results 
of these simulations with the command:
```bash
colloids-analyze analysis.yaml Run1/run.yaml Run2/run.yaml Run3/run.yaml
```
