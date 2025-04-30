# RI-measurement-operator

![language](https://img.shields.io/badge/language-Python-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

<!-- [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) -->

## Description

This branch provides a Python implementation of the RI measurement operator. The current implementation does not accommodate `w`-correction. Python version 3.10 or higher is required.

Python-based utility scripts to generate realistic Fourier sampling patterns and extract RI data from a measurement set (MS) to `.mat` file are available in `pyutils/`. Instructions are provided in `pyutils/Readme.md`.

We also provide a tutorial in the format of a Jupyter Notebook as a quick start guide about how to run the scripts in this repository. It can also be viewed [here](https://github.com/basp-group-private/RI-measurement-operator/blob/python-publish/tutorial_ri_measurement_operator_python.ipynb) .

We also provide a tutorial in the format of [Jupyter notebook](tutorial_ri_measurement_operator_python.ipynb) as a quick start guide about how to run the scripts in this repository, from setting up the environment to imaging RI measurements. It can also be viewed online [here](https://nbviewer.org/github/basp-group/RI-measurement-operator/blob/python/tutorial_ri_measurement_operator_python.ipynb).

**Contributors**: by alphabetical order, T. Chu, A. Dabbech .

## Installation

Clone the current repository

```bash
git clone -b python https://github.com/basp-group/RI-measurement-operator.git
```

## Dependencies

The code relies on external NUFFT Python libraries, the default is pytorch-finufft, using the "exponential of semicircle" kernel, as described in

> A. H. Barnett, J. Magland, and L. af Klinteberg, A parallel nonuniform fast Fourier transform library based on an “exponential of semicircle" kernel, _SIAM Journal on Scientific Computing_, 41(5), C479-C504, 2019.

- [pytorch-finufft](https://flatironinstitute.github.io/pytorch-finufft/).

With the desired virtual environment activated, install the packages using the command below:

```bash
pip install -r requirements.txt
```

#### GPU utilisation

If GPU (cuda) is available, the following package is required for `pytorch-finufft` to be able to utilise the GPU:

```bash
pip install cufinufft
```

### Optional NUFFT libraries

PyNUFFT and TorchKbNUFFT are optionally available, wich implements the NUFFT using Kaiser-Bessel kernel as proposed by

> J. A. Fessler and B. P. Sutton, Nonuniform Fast Fourier Transforms Using Min-Max Interpolation, _IEEE Trans. Image Process._, 51(2), 560-574, 2003.

They can be installed using the following commands:

- [PyNUFFT](https://pynufft.readthedocs.io/en/latest/).

```bash
pip install pynufft
```

- [TorchKbNufft](https://torchkbnufft.readthedocs.io/en/stable/).

```bash
pip install torchkbnufft
```

<!-- ### Pip -->

<!-- ### Conda

Create a new conda environment and install the packages from the provided `requirements_conda.yml` file:

``` bash
conda env create -f requirements_conda.yml
``` -->

<!-- If the conda command is not recognized and for more details regarding conda, read [conda_install.md](conda_install.md). -->

## Examples

Two examples of usage are provided:

1. `example_sim_measop.py` enables the simulation of the measurement operator and its adjoint from a Fourier sampling pattern.
   The script takes an input `.mat` file containing the `u`, `v`, `w` coordinates (in meter) and `frequency` (in Hz).

   From the terminal, run the command below:

   ```Python
   python3 example_sim_measop.py --config config/sim_meas_op.yaml
   ```

   The script will load the yaml file `config/sim_meas_op.yaml` with the following arguments:

   ```yaml
      data_file                  # (str) Path to the file containing u, v, w, frequency, and imweight (optional)
      nufft_pkg                  # (str) NUFFT package to use, default to 'finufft', 'tkbn' (TorchKbNUFFT) and 'pynufft' are also available
      im_size_x                  # (int) Target image dimension in the x direction
      im_size_y                  # (int) Target image dimension in the y direction
      super_resolution           # (float) Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
      data_weighting             # (bool) Default to fault as imaging weights are not applied in the raw measurement operator
      meas_op_on_gpu             # (bool) meaurement operator object on GPU
      nufft_oversampling_factor  # (float) Oversampling factor for the NUFFT, default to 2.0
      nufft_kernel_dim           # (int) Kaiser-Bessel kernel size for the NUFFT using TorchKbNUFFT and PyNUFFT, default to 7
      nufft_mode                 # (str) NUFFT interpolation mode when using TorchKbNUFFT, default to 'table', 'matrix' is also available for sparse matrix
      meas_dtype                 # (str) Precision of the measurement operator, default to 'double' (64-bit), 'single' (32-bit) is also available
   ```

   Note that the `--config` argument is compulsory. Input parameters can be overwritten in command line by adding `--` followed by the argument name and its value.

2. `example_sim_ri_data.py` enables the simulation of radio-inteferometric data from a given ground truth image and a Fourier sampling pattern.

   From the terminal, run the command below:

   ```Python
   python3 example_sim_ri_data.py --config config/sim_ri_data.yaml
   ```

   The script will load the yaml file `config/sim_ri_data.yaml` with the following arguments:

   ```yaml
   data_file                        # (str) Path to the file containing u, v, w, frequency, and imaging weight (optional)
   gdth_file                        # (str) Path to the ground truth file  (.fits)
   result_path                      # (str) Path to save the simulated data  (.mat, .fits)
   nufft_pkg                        # (str) NUFFT package to use, default to 'finufft', 'tkbn' (TorchKbNUFFT) and 'pynufft' are also available
   super_resolution                 # (float) Superrresolution facor, inferring the bandwidth of the imaged spatial Fourier domain
   target_dynamic_range_heuristic   # (float) Target dynamic range of the ground truth image used to infer the heuristic noise level (option 1)
   noise_isnr                       # (float) Input signa-to-noise ratio to infer the noise level (option 2)
   data_weighting                   # (bool) Default to fault as imaging weights are not applied in the raw measurement operator
   weight_type                      # (str) Type of imaging weights, default to 'briggs', 'uniform' and 'natural' are also available
   weight_robustness                # (float) Robustness parameter for the Briggs imaging weights, default to 0.0
   meas_op_on_gpu                   # (bool) meaurement operator object on GPU
   nufft_oversampling_factor        # (float) Oversampling factor for the NUFFT, default to 2.0
   nufft_kernel_dim                 # (int) Kaiser-Bessel kernel size for the NUFFT using TorchKbNUFFT and PyNUFFT, default to 7
   nufft_mode                       # (str) NUFFT interpolation mode when using TorchKbNUFFT, default to 'table', 'matrix' is also available for sparse matrix
   meas_dtype                       # (str) Precision of the measurement operator, default to 'double' (64-bit), 'single' (32-bit) is also available
   ```

   Note that the `--config` argument is compulsory. Input parameters can be overwritten in command line by adding `--` followed by the argument name and its value.
