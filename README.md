# R2D2 Algorithm
![language](https://img.shields.io/badge/language-python-orange.svg)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

- [R2D2 Algorithm](#r2d2-algorithm)
  - [Description](#description)
  - [Dependencies](#dependencies)
  - [Imaging](#imaging)
    - [Input files](#input-files)
      - [Trained DNN series](#trained-dnn-series)
      - [Data (measurement) file](#data-measurement-file)
      - [Groundtruth file](#groundtruth-file)
    - [Usage and example](#usage-and-example)
  - [Training](#training)

## Description
The R2D2 algorithm takes a hybrid structure between a Plug-and-Play (PnP) algorithm and a learned version of the well-known Matching Pursuit algorithm. Its reconstruction is formed as a series of residual images, iteratively estimated as outputs of iteration-specific Deep Neural Networks (DNNs), each taking the previous iteration’s image estimate and associated back-projected data residual as inputs.  The R2D2 algorithm comes in two incarnations. The first uses the well-known U-Net architecture for
its DNNs, and is simply referred to as R2D2. The second uses a more advanced architecture dubbed R2D2-Net, obtained by unrolling the R2D2 algorithm itself. In reference to its nesting structure, this incarnation is referred to as R3D3. The primary application of the R2D2 algorithm is to solve large-scale high-resolution high-dynamic range inverse problems in radio astronomy, more specifically 2D planar monochromatic intensity imaging.

R2D2 is part of the [BASPLib](https://basp-group.github.io/BASPLib/R2D2.html) software suite. Users can find a comprehensive [tutorial](https://basp-group.github.io/BASPLib/tutorial_r2d2_python_v1.html) for R2D2 RI v1.0.

Please refer to the following papers:

>[1] Aghabiglou, A., Chu, C. S., Dabbech, A. & Wiaux, Y., The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy, ApJS, 273(1):3, 2024, [arXiv:2403.05452](https://arxiv.org/abs/2403.05452) | [DOI:10.3847/1538-4365/ad46f5](https://doi.org/10.3847/1538-4365/ad46f5)
>
>[2] Dabbech, A., Aghabiglou, A., Chu, C. S. & Wiaux, Y., CLEANing Cygnus A deep and fast with R2D2, ApJL, 966(2), L34, 2024, [arXiv:2309.03291](https://arxiv.org/abs/2309.03291) | [DOI:10.3847/2041-8213/ad41df](https://doi.org/10.3847/2041-8213/ad41df)
>

This repository provides a full Python implementation of the R2D2 algorithm (in its two incarnations) at both training and imaging stages. 

The full path to this repository is referred to as `$R2D2` in the rest of the documentation.

A MATLAB implementation of the R2D2 algorithm (imaging only) is available in the branch [`matlab-inference`](https://github.com/basp-group/R2D2/tree/matlab-inference).

We also provide a tutorial in the format of [Jupyter notebook](https://github.com/basp-group/R2D2-SII/blob/main/tutorial_r2d2_python.ipynb) as a quick start guide about how to run the scripts in this repository, from setting up the environment to imaging RI measurements. It can also be viewed online [here](https://basp-group.github.io/BASPLib/R2D2_tutorial.html).

## Dependencies
Python version `3.10` or higher is required. PyTorch and torchvision should be installed separately by following the instructions from the [website](https://pytorch.org/get-started/locally/) to ensure their latest version available for your CUDA version is installed. For CUDA versions older than 11.8, follow the instructions from the [website](https://pytorch.org/get-started/previous-versions/). Below is an example of the command used to install PyTorch with CUDA version 11.6:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
Other required Python packages are listed in the file [`$R2D2/requirements.txt`](requirements.txt), and can be easily installed using the command below:
   ``` bash
   pip install -r requirements.txt
   ```
## Imaging

### Input files
Both R2D2 and R3D3 series are trained to form images of size `512x512` from data acquired by the Very Large Array (VLA). The input dirty images (i.e., the back-projected data) are expected to have a pixel size corresponding to a super resolution factor of `1.5`, and to be obtained using the data-weighting scheme Briggs. To run the R2D2 algorithm, data and DNN files are required.

#### Trained DNN series 
The VLA-trained R2D2 and R3D3 DNN series are available at the DOI:[10.17861/99cbe654-5071-4625-b59d-a26c790cbeb4](https://researchportal.hw.ac.uk/en/datasets/r2d2-deep-neural-network-series-for-radio-interferometric-imaging). DNN checkpoints must to be saved in a desired path `$CHECKPOINT_DIR` and under the following filename convention.  
- For R2D2, the checkpoints of its `$I` U-Net components are structured inside `$CHECKPOINT_DIR` as follows:
  ```Python
  $CHECKPOINT_DIR'/R2D2_UNet_N1.ckpt'
  $CHECKPOINT_DIR'/R2D2_UNet_N2.ckpt'
  ..
  $CHECKPOINT_DIR'/R2D2_UNet_N'$I'.ckpt'
  ```
  
- For R3D3, the checkpoints of its `$I` R2D2-Net components (each composed of `$J` U-Net layers) are structured inside `$CHECKPOINT_DIR` as follows:
  ```Python
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N1.ckpt'
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N2.ckpt'
   ..
   $CHECKPOINT_DIR'/R3D3_R2D2Net_'$J'Layers_N'$I'.ckpt'
  ```
#### Data (measurement) file
The current code takes as input data a measurement file in ``.mat`` format containing the following fields:

 ```Matlab 
   "y"               %% vector; data (Stokes I)
   "u"               %% vector; u coordinate (in units of the wavelength)
   "v"               %% vector; v coordinate (in units of the wavelength)
   "w"               %% vector; w coordinate (in units of the wavelength)
   "nW"              %% vector; inverse of the noise standard deviation 
   "nWimag"          %% vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "frequency"       %% scalar; observation frequency
   "maxProjBaseline" %% scalar; maximum projected baseline (in units of the wavelength; formally max(sqrt(u.^2+v.^2)))
   ```

- **Notes:**
  - An example measurement file ``data_3c353.mat`` is provided in the folder [`$R2D2/data/3c353/`](data/3c353/).
  - Briggs weights are generated using the [WSClean software](https://wsclean.readthedocs.io/en/latest/) with the Briggs parameter set to `0`.
  - To extract the data file from Measurement Set Tables (MS), you can use the utility Python script [`$R2D2/ms2mat/ms2mat.py`](ms2mat/ms2mat.py). Full instructions are available in [`$R2D2/ms2mat/ReadMe`](ms2mat/README.md).

#### Groundtruth file
The groundtruth file `$GT_FILE` is in `.fits` format. The file is optional and is used to compute the reconstruction evaluation metrics. An example file `3c353_GTfits.fits` is provided in the folder [`$R2D2/data/3c353/`](data/3c353/).

### Usage and example
The R2D2 algorithm (R2D2/R3D3) can be run using the following command from the terminal, specifying the path to the configuration file in `.yaml` format, an example can be found [here](config/imaging/R2D2.yaml):
``` Python
python3 ./src/imager.py --yaml_file ./config/imaging/R2D2.yaml
```

The necessary arguments in the configuration files are listed and explained below. The final reconstructions which consist of the image estimate and associated residual dirty image (i.e., back-projected residual data) are saved in `$RESULTS_DIR`. The intermediate reconstructions (outputs of each iteration) can also be saved by using the `--save_all_outputs` argument.
``` yaml
im_dim_x: 512                # (int) Image width. 
im_dim_y: 512                # (int) Image height. 
data_file: $DATA_FILE        # (str) Path to the input .mat data file. 
output_path: $RESULTS_DIR    # (str) Path to the final fits files. 
super_resolution: 1.5        # (float) Super resolution factor. 
save_all_outputs: False      # (bool, optional) Save all intermediate outputs, otherwise only final iteration results will be saved. 
gen_nWimag: True             # (bool, optional) Generate imaging weights from the sampling pattern. 
weight_type: briggs          # (str) Type of imaging weights.
weight_robustness: 0         # (float) Briggs weighting robutness parameter.
weight_gridsize: 2           # (float) Briggs weighting grid oversampling size.
series: $INCARNATION         # (str) Incarnation of the R2D2 algorithm: "R2D2" or "R3D3". 
num_iter: $I                 # (int) Number of DNNs in the R2D2/R3D3 series 
layers: $J                   # (int) Number of network layers in the DNN architecture. Currently acceptable values 1, 3, 6. 
ckpt_path: $CHECKPOINT_DIR   # (str) Path to the directory of the DNN checkpoints. 
res_on_gpu: True             # (bool, optional) Compute residual dirty images on GPU to significantly accelerate overall imaging time. 
operator_type: $OP_TYPE      # (str, optional) NUFFT interpolation: "table" or "sparse_matrix". Default: "table" which is faster, "sparse_matrix" is relatively more accurate.
gdth_file: $GT_FILE          # (str, optional) Path to the ground truth fits file. 
target_dynamic_range: $DR    # (float, optional) Target dynamic range for the computation of the logSNR metric when the groundtruth is available. 
```
- **Notes:**
   - The parameter `layers` (`$J`) takes different values depending on the considered incarnation of the R2D2 algorithm.
     -  R2D2 series: `J=1`.
     -  R3D3 series: to use the currently trained R3D3 realizations, set `J=3` or `J=6`.

   - To run the first term in the R2D2 (respectively, R3D3) series which corresponds to the end-to-end DNN U-Net (respectively, R2D2-Net) set `num_iter` (`$I`)  to `1`.
   - The parameter `target_dyanamic_range` (`$DR`) is optional and is used to compute the logSNR metric when the groundtruth image is available.

   - Examples are provided as bash shell scripts in [`$R2D2/scripts/imager.sh`](scripts/imager.sh).

 ## Training
 Detailed instructions on the training will be available soon.
