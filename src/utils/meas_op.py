import argparse

import torch

from lib import MeasOpPSF, MeasOpPynufft, MeasOpPytorchFinufft, MeasOpTkbNUFFT


def create_meas_op(args: argparse.Namespace, data: dict, os: int = 1, device=torch.device("cpu")):
    """
    _summary_

    :param args: Namespace containing the arguments: 
    nufft_pkg, img_size, nufft_oversampling_factor, nufft_kernel_dim, 
    nufft_mode, real_flag, meas_dtype
    :type args: argparse.Namespace
    :param data: dictionary containing the keys: u, v, nW, nWimag
    :type data: dict
    :param os: oversampling of the image size, set to 2 for creating PSF with 2x FoV, defaults to 1
    :type os: int, optional
    :param device: device to run the computation, defaults to torch.device('cpu')
    :type device: torch.device, optional
    """
    match args.nufft_pkg:
        case "finufft":
            Operator = MeasOpPytorchFinufft
        case "tkbn":
            Operator = MeasOpTkbNUFFT
        case "pynufft":
            Operator = MeasOpPynufft
        case "psf":
            Operator = MeasOpPSF

    img_size = tuple(int(i * os) for i in args.img_size)
    nufft_grid_size = tuple(int(i * args.nufft_oversampling_factor) for i in img_size)
    meas_op = Operator(
        u=data["u"],
        v=data["v"],
        img_size=img_size,
        real_flag=args.real_flag,
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        grid_size=nufft_grid_size,
        num_points=args.nufft_kernel_dim,
        mode=args.nufft_mode,
        device=device,
        dtype=args.meas_dtype,
    )

    return meas_op


def create_dummy_meas_op(args: argparse.Namespace, os: int = 1, device=torch.device("cpu")):
    """
    _summary_

    :param args: _description_
    :type args: argparse.Namespace
    :param data: _description_
    :type data: dict
    """

    data = {}
    data["u"] = torch.tensor([1.0, 1.0], dtype=args.meas_dtype, device=device).view(1, 1, -1)
    data["v"] = torch.tensor([1.0, 1.0], dtype=args.meas_dtype, device=device).view(1, 1, -1)
    data["nW"] = torch.tensor([1.0, 1.0], dtype=args.meas_dtype, device=device).view(1, 1, -1)
    data["nWimag"] = torch.tensor([1.0, 1.0], dtype=args.meas_dtype, device=device).view(1, 1, -1)

    return create_meas_op(args, data, os, device)
