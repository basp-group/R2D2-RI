import argparse
import os
import platform
from enum import Enum
from typing import Optional

import psutil
import torch
import yaml
from pydantic import BaseModel, Field, FilePath, model_validator


def parse_yaml_file():
    """
    Parse a YAML file containing configuration arguments and return the parsed arguments.

    :return: parsed argument with yaml file path.
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to yaml file containing all the arguments."
    )

    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--gdth_file", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)

    parser.add_argument("--nufft_pkg", choices=["finufft", "tkbn", "pynufft"], default=None)
    parser.add_argument("--super_resolution", type=float, default=None)
    parser.add_argument("--image_pixel_size", type=float, default=None)
    parser.add_argument("--nufft_oversampling_factor", type=float, default=None)
    parser.add_argument("--nufft_kernel_dim", type=int, default=None)
    parser.add_argument("--nufft_mode", choices=["table", "matrix"], default=None)
    parser.add_argument("--real_flag", action="store_true", default=None)
    parser.add_argument("--meas_op_on_gpu", action="store_true", default=None)
    parser.add_argument("--meas_dtype", choices=["single", "double"], default=None)

    parser.add_argument("--data_weighting", action="store_true", default=None)
    parser.add_argument("--load_weight", action="store_true", default=None)
    parser.add_argument("--weight_type", choices=["uniform", "briggs"], default=None)
    parser.add_argument("--weight_gridsize", type=float, default=None)
    parser.add_argument("--weight_robustness", type=float, default=None)

    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--ncpus", type=int, default=None)

    parser.add_argument("--target_dynamic_range_heuristic", type=float, default=None)
    parser.add_argument("--noise_isnr", type=float, default=None)

    parser.add_argument("--im_dim_x", type=int, default=None)
    parser.add_argument("--im_dim_y", type=int, default=None)

    args = parser.parse_args()
    return args


class NufftMode(str, Enum):
    table = "table"
    matrix = "matrix"


class DType(str, Enum):
    single = "single"
    double = "double"


class WeightType(str, Enum):
    uniform = "uniform"
    briggs = "briggs"
    natural = "natural"


class NufftPkgEnum(str, Enum):
    finufft = "finufft"
    tkbn = "tkbn"
    pynufft = "pynufft"


class Params(BaseModel):
    # i/o
    data_file: FilePath

    # measurement operator
    nufft_pkg: NufftPkgEnum = NufftPkgEnum.finufft
    super_resolution: float = 1.0
    image_pixel_size: float = None
    nufft_oversampling_factor: float = 2.0
    nufft_kernel_dim: int = 7
    nufft_mode: NufftMode = NufftMode.table
    real_flag: bool = True
    meas_op_on_gpu: bool = False
    meas_dtype: DType = DType.double

    # imaging weights
    data_weighting: bool = False
    load_weight: bool = False
    weight_type: WeightType = WeightType.briggs
    weight_gridsize: float = 2.0
    weight_robustness: float = 0.0

    # misc
    verbose: bool = True
    ncpus: int = None

    class Config:
        extra = "allow"
        validate_assigment = True


class MeasOpParams(Params):
    im_dim_x: int = 512
    im_dim_y: int = 512


class SimRIParams(Params):
    # i/o
    gdth_file: FilePath
    output_path: str = None

    # simulation
    target_dynamic_range_heuristic: Optional[float] = Field(default=None, ge=0)
    noise_isnr: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    @classmethod
    def _validate_noise(cls, values):
        if sum([bool(values.target_dynamic_range_heuristic), bool(values.noise_isnr)]) != 1:
            raise ValueError("Exactly one of target_dynamic_range_heuristic or noise_isnr must be specified")
        return values


def set_args(args):
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

    if args.meas_dtype == DType.single:
        args.__setattr__("meas_dtype", torch.float)
    elif args.meas_dtype == DType.double:
        args.__setattr__("meas_dtype", torch.double)

    args.device = (
        torch.device("cuda") if args.meas_op_on_gpu and torch.cuda.is_available() else torch.device("cpu")
    )

    # args.model_config['extra'] = 'forbid'
    return args




def parse_args_meas_op(config: str = None):
    if config is None:
        args_yaml = parse_yaml_file()
    else:
        args_yaml = argparse.Namespace(config=config)
    with open(args_yaml.config, "r") as file:
        yaml_loaded = yaml.safe_load(file)
    args = MeasOpParams(**yaml_loaded)
    for k, v in args_yaml.__dict__.items():
        if v is not None:
            args.__setattr__(k, v)
    args.__setattr__("img_size", (args.im_dim_x, args.im_dim_y))
    args.__setattr__(
        "nufft_grid_size",
        (
            int(args.im_dim_x * args.nufft_oversampling_factor),
            int(args.im_dim_y * args.nufft_oversampling_factor),
        ),
    )
    args = set_args(args)
    return args


def parse_args_sim_ri_data(config: str = None):
    if config is None:
        args_yaml = parse_yaml_file()
    else:
        args_yaml = argparse.Namespace(config=config)
    with open(args_yaml.config, "r") as file:
        yaml_loaded = yaml.safe_load(file)
    args = SimRIParams(**yaml_loaded)
    for k, v in args_yaml.__dict__.items():
        if v is not None:
            args.__setattr__(k, v)
    args = set_args(args)

    os.system(f"mkdir -p {args.result_path}")
    print(f"INFO: Output path: {args.result_path}")
    return args
