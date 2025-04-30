#!/usr/bin/env python3
# Author: Taylor C.
"""
Functions to parse the configuration file in .yaml format, and validate all
arguments set from the configuration file using Pydantic model.
"""

###################################################
# imports

import argparse
import os
import pathlib
import platform
from enum import Enum, IntEnum
from typing import List, Optional, Union

import psutil
import torch
import yaml
from pydantic import BaseModel, ConfigDict, DirectoryPath, FilePath, ValidationError, field_validator

###################################################


def parse_config():
    """
    Parse a YAML file containing configuration arguments and return the parsed arguments.

    :return: parsed argument with yaml file path.
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to yaml file containing all the arguments."
    )

    parser.add_argument("--src_name", type=str, default=None)
    parser.add_argument("--gdth_file", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--save_all_outputs", action="store_true", default=None)
    parser.add_argument("--data_init", action="store_true", default=None)

    parser.add_argument("--meas_op_on_gpu", action="store_true", default=None)
    parser.add_argument("--super_resolution", type=float, default=None)
    parser.add_argument("--im_dim_x", type=int, default=None)
    parser.add_argument("--im_dim_y", type=int, default=None)

    parser.add_argument("--weight_type", choices=["briggs", "uniform", "none"], default=None)
    parser.add_argument("--weight_gridsize", type=float, default=None)
    parser.add_argument("--weight_robustness", type=float, default=None)

    parser.add_argument("--series", choices=["R2D2", "R3D3"], default=None)
    parser.add_argument("--num_iter", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_realisations", type=int, default=None)
    parser.add_argument("--architecture", choices=["unet", "uwdsr", "U-Net", "U-WDSR"], default=None)

    parser.add_argument("--nufft_pkg", choices=["tkbn", "finufft", "pynufft", "psf"], default=None)
    parser.add_argument("--nufft_oversampling_factor", type=float, default=None)
    parser.add_argument("--nufft_kernel_dim", type=int, default=None)
    parser.add_argument("--nufft_mode", choices=["table", "matrix"], default=None)
    parser.add_argument("--real_flag", action="store_true", default=None)

    parser.add_argument("--target_dynamic_range", type=float, default=None)

    parser.add_argument("--prune", action="store_true", default=None)
    parser.add_argument("--sigma_res_tol", type=float, default=None)
    
    

    args = parser.parse_args()
    return args


###################################################


class SeriesEnum(str, Enum):
    """
    Enum class for the series of R2D2 and R3D3.
    """

    R2D2 = "R2D2"
    R2D2Net = "R2D2Net"
    R3D3 = "R3D3"


class LayersEnum(IntEnum):
    """
    Enum class for the number of layers in the R2D2Net for the R3D3 series.
    """

    one = 1
    three = 3
    six = 6


class NufftMode(str, Enum):
    table = "table"
    matrix = "matrix"


class WeightTypeEnum(str, Enum):
    """
    Enum class for the type of weighting to generate.
    """

    briggs = "briggs"
    uniform = "uniform"
    none = "none"


class ArchitectureEnum(str, Enum):
    """
    Enum class for the type of architecture to use.
    """

    unet = "unet"
    uwdsr = "uwdsr"
    UNet = "U-Net"
    UWDSR = "U-WDSR"


class InputOrderEnum(str, Enum):
    res_rec = "res_rec"
    rec_res = "rec_res"


class NufftPkgEnum(str, Enum):
    tkbn = "tkbn"
    finufft = "finufft"
    pynufft = "pynufft"
    psf = "psf"


class CommonArgs(BaseModel):
    """
    Pydantic model class containing common arguments for different tasks.
    """

    model_config = ConfigDict()

    # measurement operator
    nufft_pkg: NufftPkgEnum = NufftPkgEnum.finufft
    super_resolution: float = 1.5
    image_pixel_size: float = None
    nufft_oversampling_factor: float = 2.0
    nufft_kernel_dim: int = 7
    nufft_mode: NufftMode = NufftMode.table
    real_flag: bool = True
    meas_op_on_gpu: bool = False

    # weighting
    data_weighting: bool = True
    natural_weight: bool = True
    weight_type: WeightTypeEnum = WeightTypeEnum.briggs
    weight_gridsize: float = 2
    weight_robustness: float = 0.0

    # calibraiton
    load_die: bool = False

    # miscellaneous
    verbose: bool = True
    ncpus: int = None

    class Config:
        extra = "allow"
        validate_assigment = True


###################################################


class ImagingArgs(CommonArgs):
    """
    Pydantic model class containing arguments specific to R2D2 imaging.
    """

    # algorithm
    num_iter: int
    series: SeriesEnum = SeriesEnum.R2D2
    layers: LayersEnum = LayersEnum.one
    data_init: bool = False

    # network architecture
    architecture: ArchitectureEnum = ArchitectureEnum.unet
    num_chans: int = 64
    num_pools: int = 4
    drop_prob: float = 0.0
    input_order: InputOrderEnum = InputOrderEnum.rec_res

    # i/o
    ckpt_path: DirectoryPath
    ckpt_realisations: int = 1 # if > 1, epistemic uncertainty quantification will be computed
    data_file: FilePath
    output_path: str = "./results/"
    gdth_file: Optional[FilePath] = None
    save_all_outputs: bool = False
    src_name: str = None

    # metrics
    target_dynamic_range: float = None

    # measurement operator
    im_dim_x: int = 512
    im_dim_y: int = 512
    
    # pruning
    prune: bool = False
    sigma_res_tol: float = 1e-4
    

    # validation
    @field_validator("layers")
    @classmethod
    def _check_layers(cls, v, values):
        if values.data["series"] == SeriesEnum.R2D2:
            if v != LayersEnum.one.value:
                # raise ValueError('R2D2 series must have only one layer.')
                print("WARNING: R2D2 series must have only one layer, this will be set automatically.")
                v = LayersEnum.one
        elif values.data["series"] == SeriesEnum.R3D3:
            if v == LayersEnum.one.value:
                raise ValueError(
                    "R3D3 series must have more than one layer, please change `layers` value in config file!"
                )
        return v

    @field_validator("data_file")
    @classmethod
    def _check_data_file(cls, v):
        assert str(v).endswith(".mat") or str(v).endswith(
            ".fits"
        ), "The provided data_file format is not currently supported. (only .fits or .mat are supported)"
        return v

    @field_validator("save_all_outputs", mode="before")
    @classmethod
    def _check_save_all_outputs(cls, v):
        if v is None:
            return False
        else:
            return v

    @field_validator("target_dynamic_range", mode="before")
    @classmethod
    def _check_target_dynamic_range(cls, v):
        if v is None:
            return 0.0
        else:
            return v

    @field_validator("series", "layers")
    @classmethod
    def _return_value(cls, v):
        return v.value


# Pytorch lightning (training/ testing) related argparse functions


def parse_args_pl():
    """Parse all required and optional arguments from command line.

    Returns
    -------
    _ArgumentParser
        Arguments parsed from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")

    # Model specific hyperparameters
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_pools", type=int, default=4, help="Number of U-Net pooling layers")
    parser.add_argument("--drop_prob", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--num_chans", type=int, default=64, help="Number of U-Net channels")
    parser.add_argument("--num_chans_in", type=int, default=2, help="Number of input channels")
    parser.add_argument("--num_chans_out", type=int, default=1, help="Number of output channels")
    parser.add_argument("--batch_size", default=1, type=int, help="Mini batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr-step-size", type=int, default=1000, help="Period of learning rate decay")
    parser.add_argument(
        "--lr-gamma", type=float, default=0.0, help="Multiplicative factor of learning rate decay"
    )
    parser.add_argument("--gradient_clip_val", type=float, default=1e-2)
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Strength of weight decay regularization"
    )
    parser.add_argument(
        "--positivity", action="store_true", help="If True, enforce positivity constraint on the output"
    )
    parser.add_argument("--series", choices=["R2D2", "R3D3"], required=True, help="series to train")
    parser.add_argument(
        "--layers", type=int, default=1, help="If > 1, R2D2-Net would be used instead of U-Net in R2D2"
    )
    parser.add_argument(
        "--num_iter", type=int, required=True, help="Iteration number in the R2D2/ R3D3 series to train"
    )

    # Resource related arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus in each node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of gpu nodes to be used")
    parser.add_argument(
        "--exp_dir",
        type=pathlib.Path,
        default="experiments",
        help="Path where the checkpoint and hyparameters will be saved",
    )
    parser.add_argument("--exp", type=str, help="Name of the experiment")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If True, resume the training from a previous model checkpoint. ",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, help="Path to pre-trained model. Use with --mode test"
    )

    # Dataset related arguments
    parser.add_argument("--data_path", type=pathlib.Path, required=True, help="Path to the datasets")
    parser.add_argument("--im_dim_x", type=int, default=512, help="Image dimension in x direction.")
    parser.add_argument("--im_dim_y", type=int, default=512, help="Image dimension in y direction.")
    parser.add_argument(
        "--scname_train", type=str, default="trainingset_fullnumpy", help="GT training set folder name"
    )
    parser.add_argument(
        "--scname_val", type=str, default="validationset_fullnumpy", help="GT validation set folder name"
    )
    parser.add_argument(
        "--scname_test", type=str, default="testset_fullnumpy", help="GT test set folder name"
    )
    parser.add_argument("--rec_ext", type=str, default="_recN1", help="reconstruction input folder extension")
    parser.add_argument(
        "--rec_file_ext", type=str, default="_rec", help="reconstruction image filename ending"
    )
    parser.add_argument("--res_ext", type=str, default="_resN1", help="residual input folder extension")
    parser.add_argument("--dirty_file_ext", type=str, default="_dirty", help="dirty image filename ending")
    parser.add_argument(
        "--res_file_ext",
        type=str,
        required=True,
        help="residual image filename ending, e.g. _dirty for N1, _res for Ni",
    )
    parser.add_argument(
        "--gdth_file_ext", type=str, default="_gdth", help="ground truth image filename ending"
    )
    parser.add_argument(
        "--mat_ext",
        type=str,
        default=None,
        help="ending of the folder with .mat files containing the dynamic range of the exponentiated image (1/a_expo).",
    )
    parser.add_argument(
        "--mat_file_ext", type=str, default=None, help="filename ending of the files in mat_ext."
    )

    # R2D2Net/ R3D3 specific arguments
    parser.add_argument("--dirty_ext", type=str, default="_dirty", help="dirty image folder extension")
    parser.add_argument("--PSF_path", type=str, help="path to PSF files for unrolled R2D2Net")
    parser.add_argument("--PSF_file_ext", type=str, default="_PSF", help="PSF filename ending")

    # imaging related arguments
    parser.add_argument(
        "--rec2_ext", type=str, default="_recN1", help="reconstruction output folder extension"
    )
    parser.add_argument("--save_output", action="store_true", help="Save output")
    return parser.parse_args()


###################################################


class RIDataSimulationArgs(CommonArgs):
    """
    Pydantic model class containing common arguments for different tasks.
    """

    model_config = ConfigDict()

    seed: Union[int, str] = 1377

    # i/o
    gdth_path: DirectoryPath = None
    output_path: str
    uv_path: DirectoryPath
    dataset: str = "test"
    save_vis: bool = False
    save_PSF: bool = False
    save_dirty: bool = False  # option to save dirty images when generating visibilties

    # noise
    sigma_range_min: float = 0.0
    sigma_range_max: float = 0.0
    multi_noise: bool = False

    # exponentiation
    sigma0: float = 0.0
    expo: bool = False

    # miscellaneous
    uv_random: bool = False

    # validation
    @field_validator("seed")
    @classmethod
    def _check_seed(cls, v):
        if type(v) == int:
            if v == 0:
                print("WARNING: Seed value 0 is not recommended, this will be set to 1377.")
                v = 1377
            assert v > 0, "Seed value must be positive."
        elif type(v) == "str":
            assert (
                v == "uv"
            ), "Only string value `uv` is accepted for seed, which will use the uv_id in the filename of the uv file as seed."
        return v

    @field_validator("sigma_range_min", "sigma_range_max")
    @classmethod
    def _check_sigma_range(cls, v):
        if v <= 0:
            raise ValueError("Sigma range must be non-negative and non-zero.")
        return v

    @field_validator("sigma_range_max")
    @classmethod
    def _check_sigma_range_max(cls, v, values):
        if v < values.data["sigma_range_min"]:
            raise ValueError("Sigma range min must be less than sigma range max.")
        return v

    @field_validator("expo")
    @classmethod
    def _check_expo(cls, v, values):
        if v:
            assert values.data["sigma0"] > 0, "Sigma0 must be positive for exponentiation."
            assert (
                values.data["sigma0"] > values.data["sigma_range_max"]
            ), "Sigma0 must be greater than sigma range max for exponentiation."
        return v


###################################################


class ComputeResidualArgs(CommonArgs):
    """
    Pydantic model class containing common arguments for different tasks.
    """

    model_config = ConfigDict()

    # i/o
    output_path: str
    uv_path: DirectoryPath
    # residual specific:
    mat_path: DirectoryPath
    dirty_path: DirectoryPath
    rec_path: DirectoryPath
    output_res_path_suffix: str
    prune: bool = False
    dataset: str = "test"


###################################################


def set_common_args(args):

    if platform.system() != "Darwin":  # not on macOS
        avail_cpus = len(psutil.Process().cpu_affinity())
        if args.ncpus is not None and args.ncpus >= 1:
            request_cpus = min(avail_cpus, int(args.ncpus))
            torch.set_num_threads(request_cpus)
            if args.verbose:
                print(f"INFO: avaiable cpus {avail_cpus}, request cpus {request_cpus}")
        else:
            torch.set_num_threads(avail_cpus)
            if args.verbose:
                print(f"INFO: avaiable cpus {avail_cpus}")

    args.__setattr__("meas_dtype", torch.double)

    args.device = (
        torch.device("cuda") if args.meas_op_on_gpu and torch.cuda.is_available() else torch.device("cpu")
    )

    return args


def parse_args_ri_data_simulation():
    args_yaml = parse_config()
    with open(args_yaml.config, "r") as file:
        args_yaml.__dict__.update(yaml.safe_load(file))
    args = RIDataSimulationArgs(**args_yaml.__dict__)
    return set_common_args(args)


def parse_args_compute_residual():
    args_yaml = parse_config()
    with open(args_yaml.config, "r") as file:
        args_yaml.__dict__.update(yaml.safe_load(file))
    args = ComputeResidualArgs(**args_yaml.__dict__)
    return set_common_args(args)


def parse_args_imaging():
    """
    Parses the arguments for imaging from a YAML file and updates the argument object with additional parameters.

    This function reads a YAML file specified by the `--config` argument, updates the argument object with the parameters
    specified in the YAML file, and sets additional parameters for imaging. It also performs some checks and prints
    information messages.

    :return: The updated argument object.
    :rtype: argparse.Namespace

    :raises AssertionError: If the `data_file` argument does not end with '.mat' or '.fits'.
        If the `series` argument is 'R3D3' and `layers` is not greater than 1.
    """
    args_yaml = parse_config()
    with open(args_yaml.config, "r") as file:
        yaml_loaded = yaml.safe_load(file)
    args = ImagingArgs(**yaml_loaded)
    for k, v in args_yaml.__dict__.items():
        if v is not None:
            args.__setattr__(k, v)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.architecture in ["unet", "U-Net", "UNet"]:
        args.architecture = ArchitectureEnum.unet
    elif args.architecture in ["uwdsr", "U-WDSR", "UWDSR"]:
        args.architecture = ArchitectureEnum.uwdsr

    args.img_size = (args.im_dim_x, args.im_dim_y)
    args.nufft_grid_size = tuple(int(i * args.nufft_oversampling_factor) for i in args.img_size)
    return set_common_args(args)
