# R2D2 RI V2.0 Algorithm

![language](https://img.shields.io/badge/language-python-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

- [R2D2 Algorithm](#r2d2-algorithm)
  - [Description](#description)
  - [Dependencies](#dependencies)
    - [Cloning the project](#cloning-the-project)
    - [Python packages](#python-packages)
    - [NUFFT packages](#nufft-packages)
  - [Input files](#input-files)
    - [VLA-trained DNN series](#vla-trained-dnn-series)
    - [Data (measurement) file](#data-measurement-file)
    - [Groundtruth file](#groundtruth-file)
  - [Usage and Example](#usage-and-example)
    - [Imaging / Test stage](#imaging--test-stage)
    - [Training](#training)

## Description

The R2D2 algorithm takes a hybrid structure between a Plug-and-Play (PnP) algorithm and a learned version of the well-known "Matching Pursuit" algorithm. Its reconstruction is formed as a series of residual images, iteratively estimated as outputs of Deep Neural Networks (DNNs) taking the previous iteration’s image estimate and associated data residual as inputs. R2D2's primary application is to solve large-scale, high-resolution, high-dynamic range inverse problems for RI in radio astronomy, more specifically, 2D planar monochromatic intensity imaging. This repository corresponds to the latest version of the R2D2 algorithm (v2.0). Earlier versions (v1.0, both MATLAB and Python) are available as separate branches.
R2D2 is part of the [BASPLib](https://basp-group.github.io/BASPLib/R2D2.html) software suite. Users can find a comprehensive R2D2 RI v2.0 [tutorial](https://basp-group.github.io/BASPLib/tutorial_r2d2_python_v2.html) and [benchmarking webpage](https://basp-group.github.io/BASPLib/benchmarking.html) where R2D2 is evaluated against state-of-the-art algorithms for RI imaging.
Please refer to the following papers:

<!-- [arXiv:2403.05452](https://arxiv.org/abs/2403.05452) -->
> [1] Aghabiglou, A., Chu, C. S., Tang, C., Dabbech, A. & Wiaux, Y., Towards a robust R2D2 paradigm for radio-interferometric imaging: revisiting DNN training and architecture, submitted to ApJS, [arXiv:2503.02554](https://arxiv.org/abs/2503.02554)
>
> [2] Aghabiglou, A., Chu, C. S., Dabbech, A. & Wiaux, Y., R2D2 image reconstruction with model uncertainty quantification in radio astronomy, EUSIPCO, 2024, [arXiv:2403.18052](https://arxiv.org/abs/2403.18052) | [DOI:10.23919/EUSIPCO63174.2024.10715010](https://doi.org/10.23919/EUSIPCO63174.2024.10715010)
>
> [3] Aghabiglou, A., Chu, C. S., Dabbech, A. & Wiaux, Y., The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy, ApJS, 273(1), 3, 2024, [arXiv:2403.05452](https://arxiv.org/abs/2403.05452) | [DOI:10.3847/1538-4365/ad46f5](https://doi.org/10.3847/1538-4365/ad46f5)
>
> [4] Dabbech, A., Aghabiglou, A., Chu, C. S. & Wiaux, Y., CLEANing Cygnus A deep and fast with R2D2, ApJL, 966(2), L34, 2024, [arXiv:2309.03291](https://arxiv.org/abs/2309.03291) | [DOI:10.3847/2041-8213/ad41df](https://doi.org/10.3847/2041-8213/ad41df)

This repository provides a full Python implementation of the R2D2 algorithm at both training and imaging stages.

## Dependencies

### Cloning the project

Clone the repository with the required submodule using one of the commands below:

- using a password-protected SSH key:

  ```bash
  git clone --recurse-submodules git@github.com:basp-group/R2D2-RI.git
  ```

- using the web URL:

  ```bash
  git clone --recurse-submodules https://github.com/basp-group/R2D2-RI.git
  ```

### Python packages

All required Python packages are listed in the [requirements](requirements.txt) file. Python version `3.10` or higher is required.

Install the packages using the command below:

```bash
pip install -r requirements.txt
```

### NUFFT packages

The data fidelity layers are computed using the Non-Uniform Fast Fourier Transform (NUFFT) operator, which is implemented in the python branch of the submodule [RI-measurement-operator](https://github.com/basp-group/RI-measurement-operator/tree/python). The default NUFFT package used is [pytorch-finufft](https://flatironinstitute.github.io/pytorch-finufft/), which is included in the [requirements](requirements.txt) file. If GPU with cuda driver is available, an additional package is required for `pytorch-finufft` to be able to utilise the GPU, using the command:

```bash
pip install cufinufft
```

PyNUFFT and TorchKbNUFFT are optionally available, which can be installed using the following commands:

- [PyNUFFT](https://pynufft.readthedocs.io/en/latest/).

```bash
pip install pynufft
```

- [TorchKbNufft](https://torchkbnufft.readthedocs.io/en/stable/).

```bash
pip install torchkbnufft
```

## Input files

The available R2D2 series are trained specifically for images of size `512x512`. To run the R2D2 algorithm, data and DNN files are required.

### VLA-trained DNN series

- The VLA-trained R2D2 (U-Net & U-WDSR) DNN series proposed in [1] are available at the DOI:[10.17861/e3060b95-4fe6-4b61-9f72-d77653c305bb](https://researchportal.hw.ac.uk/en/datasets/robust-r2d2-dnn-series-for-monochromatic-intensity-imaging-with-v). The series support varying pixel-resolutions, corresponding to super resolution factors in the interval [1.5, 2.5]. They also support the Briggs data-weighting scheme with varying robustness parameter in the interval [-1, 1]. Five realisations of both series are trained to enable epistemic uncertainty quantification.

DNN checkpoints need to be saved in a desired path `$CHECKPOINT_DIR`.

- For R2D2 with U-Net as a core architecture, the DNN checkpoints are structured inside `$CHECKPOINT_DIR` as follows:

  ```bash
  $CHECKPOINT_DIR'/R2D2_UNet_N1.ckpt'
  $CHECKPOINT_DIR'/R2D2_UNet_N2.ckpt'
  ..
  $CHECKPOINT_DIR'/R2D2_UNet_N'$I'.ckpt'
  ```

- For R2D2 with U-WDSR as a core architecture, the DNN checkpoints are structured inside `$CHECKPOINT_DIR` as follows:

  ```bash
  $CHECKPOINT_DIR'/R2D2_UWDSR_N1.ckpt'
  $CHECKPOINT_DIR'/R2D2_UWDSR_N2.ckpt'
  ..
  $CHECKPOINT_DIR'/R2D2_UWDSR_N'$I'.ckpt'
  ```


### Data (measurement) file

The current code takes as input data a measurement file in `.mat` format containing the following fields:

```matlab
  "y"               %% vector; data (Stokes I)
  "u"               %% vector; u coordinate (in units of the wavelength)
  "v"               %% vector; v coordinate (in units of the wavelength)
  "w"               %% vector; w coordinate (in units of the wavelength)
  "nW"              %% vector; inverse of the noise standard deviation
  "nWimag"          %% vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
  "frequency"       %% scalar; observation frequency
  "maxProjBaseline" %% scalar; maximum projected baseline (in units of the wavelength; formally  max(sqrt(u**2+v**2)))
```

- **Notes:**
  - To extract the data file from Measurement Set Tables (MS), you can use the utility Python script `pyxisMs2mat/pyxis_ms2mat.py`. Full instructions are available
  in the [ReadMe File](pyxisMs2mat/ReadMe.md).
  - An example measurement file `data_3c353.mat` is provided in the folder `data/3c353/`.

### Groundtruth file

The groundtruth file `$GT_FILE` is in `.fits` format. The file is optional, used to compute the reconstruction evaluation metrics.

## Usage and Example

### Imaging / Test stage

The R2D2 algorithm can be run using the following command in the terminal. The final reconstructions which consist of the image estimate and associated residual dirty image are saved 
in `$RESULTS_DIR`. The intermediate reconstructions can also be saved using the argument `--save_all_outputs`. 

```bash
python3 ./src/imager.py --config $yaml_file
```

Configuration file `$yaml_file` is in `.yaml` format. Examples for each of the R2D2 variants can be found in [config/imaging](config/imaging), and is structured as follows:

```yaml
# i/o
gdth_file: # (str, optional) Path to the ground truth fits file, used for computaiton of SNR and logSNR metric.
data_file: # (str) Path to the input .mat data file.
output_path: # (str) Path to the final image files.
src_name: # (str, optional) Source name, the output folder will be named as.
save_all_outputs: # (bool, optional) Save all intermediate outputs, otherwise only final iteration results
will be saved, default to False.

# measurement operator
nufft_pkg: # (str) NUFFT package to use, default to 'finufft', 'tkbn' (TorchKbNUFFT) and 'pynufft' are also available
meas_op_on_gpu: # (bool, optional) Compute residual dirty images on GPU to significantly accelerate overall imaging
time, default to False.
super_resolution: # (float) Super resolution factor.
im_dim_x: # (int) Image width.
im_dim_y: # (int) Image height.

# algorithm
num_iter: # (int) Number of DNNs in the R2D2 series
ckpt_path: # (str) Path to the directory of the DNN checkpoints.
ckpt_realisations: # (int) Number of realisations of the DNN series, if larger than 1, epistemic uncertainty
quantification will be computed automatically.
architecture: # (str, optional) DNN architecture, choose from ['unet', 'uwdsr'], default to 'unet'.
num_chans: # (int, optional) Number of channels in the DNN architecture, 32 for the pre-trained U-Nets and 64 for
the pre-trained U-WDSRs, default to 64.

target_dynamic_range: # (float, optional) Target dynamic range for computation of logSNR metric and
stadnard-over-mean image for epistemic uncertainty quantification, if not specified, the reciprocal of the heuristic 1/sqrt(2L)
will be used automatically, where L is the spectral norm of the measurement operator.
```

- **Notes:**

  - All abovementioned arguments can be set through command line arguments, e.g.`--data_file data/3c353/data_3c353.mat`, `--output_path results`, etc. Command line arguments will overwrite the arguments in the configuration file.
  - Examples to run each R2D2 incarnation are provided as bash shell scripts in [examples](examples), along with their corresponding configuration files in [config/imaging](config/imaging).

### Training

The instructions on training will be available soon.
