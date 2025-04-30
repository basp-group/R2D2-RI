import argparse

import numpy as np
import torch

from pysrc.measOperator import MeasOpDTFT, MeasOpPynufft, MeasOpPytorchFinufft, MeasOpTkbNUFFT
from pysrc.utils import load_data_to_tensor


def test_adjoint_op(meas_op, img_size, M, seed=1377):
    np.random.seed(seed)
    x = torch.randn(1, 1, *img_size, dtype=torch.float64)
    y = (
        torch.randn(1, 1, M, dtype=torch.complex128) + 1j * torch.randn(1, 1, M, dtype=torch.complex128)
    ) / np.sqrt(2.0)
    
    if "finufft" in str(meas_op.__class__.__name__).lower():
        p1 = torch.sum(y.conj() * meas_op.forward_op(x))
        p2 = torch.sum((meas_op.adjoint_op(y) * x))
    else:
        p1 = torch.sum(y.conj() * meas_op.forward_op(x))
        p2 = torch.sum((meas_op.adjoint_op(y).conj() * x))

    rel_diff = abs(p1 - p2) / abs(p1)
    if rel_diff < 1e-10:
        print("Adjoint operator test passed")
    else:
        print("Adjoint operator test failed")
        print(f"Relative difference: {rel_diff}")


def rand_data(data, M, seed=1377):
    np.random.seed(seed)
    rand_idx = np.random.choice(np.arange(data["u"].shape[-1]), M, replace=False)
    for k in ["u", "v", "w"]:
        data[k] = data[k][..., rand_idx]
    return data


def test_meas_op_dtft(data, img_size, M, device=torch.device("cpu"), seed=1377):
    max_proj_baseline = data["max_proj_baseline"]
    spatial_bandwidth = 2 * max_proj_baseline
    image_pixel_size = (180.0 / torch.pi) * 3600.0 / (data["super_resolution"] * spatial_bandwidth)
    data = rand_data(data, M, seed)
    meas_op = MeasOpDTFT(
        u=data["u"],
        v=data["v"],
        pixel_size=image_pixel_size,
        img_size=img_size,
        real_flag=False,
        dtype=torch.float64,
        device=device,
    )
    print("Adjoint test for DTFT:")
    test_adjoint_op(meas_op, img_size, M, seed)
    return meas_op.get_psf().real.squeeze()


def test_meas_op_nufft(operator, data, img_size, M, Jd, device=torch.device("cpu"), seed=1377):
    data = rand_data(data, M, seed)
    meas_op = operator(
        u=data["u"],
        v=data["v"],
        img_size=img_size,
        kernel_dim=Jd,
        grid_size=tuple(2 * np.array(img_size)),
        real_flag=False,
        dtype=torch.float64,
        device=device,
    )
    op_name = str(meas_op.__class__).split(".")[-1].split("MeasOp")[-1][:-2]
    print(f"Adjoint test for {op_name}:")
    test_adjoint_op(meas_op, img_size, M, seed)
    return meas_op.get_psf().real.squeeze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size_x", type=int, default=64)
    parser.add_argument("--img_size_y", type=int, default=64)
    parser.add_argument("--M", type=int, default=int(1e4))
    parser.add_argument("--Jd", type=int, default=7)
    args = parser.parse_args()

    img_size = (args.img_size_x, args.img_size_y)
    M = args.M
    Jd = args.Jd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_radian = load_data_to_tensor(
        uv_file_path="data/test.mat",
        super_resolution=1.5,
        data_weighting=False,
        img_size=img_size,
        uv_unit="radians",
        device=device,
        verbose=True,
    )
    data_lam = load_data_to_tensor(
        uv_file_path="data/test.mat",
        super_resolution=1.5,
        data_weighting=False,
        img_size=img_size,
        uv_unit="wavelength",
        device=device,
        verbose=True,
    )

    psf_dtft = test_meas_op_dtft(data_lam, img_size, M, device)
    psf_tkbn = test_meas_op_nufft(
        operator=MeasOpTkbNUFFT, data=data_radian, img_size=img_size, M=M, Jd=Jd, device=device
    )
    psf_finufft = test_meas_op_nufft(
        operator=MeasOpPytorchFinufft, data=data_radian, img_size=img_size, M=M, Jd=Jd, device=device
    )
    psf_pynufft = test_meas_op_nufft(
        operator=MeasOpPynufft, data=data_radian, img_size=img_size, M=M, Jd=Jd, device=device
    )

    print("Comparing point spread functions:")
    psf_rel_diff_tkbn = torch.linalg.norm(psf_dtft - psf_tkbn) / torch.linalg.norm(psf_dtft)
    print(f"Relative difference between DTFT and TkBN: {psf_rel_diff_tkbn.item():.4e}")
    psf_rel_diff_finufft = torch.linalg.norm(psf_dtft - psf_finufft) / torch.linalg.norm(psf_dtft)
    print(f"Relative difference between DTFT and Pytorch-Finufft: {psf_rel_diff_finufft.item():.4e}")
    psf_rel_diff_pynufft = torch.linalg.norm(psf_dtft - psf_pynufft) / torch.linalg.norm(psf_dtft)
    print(f"Relative difference between DTFT and PyNUFFT: {psf_rel_diff_pynufft.item():.4e}")
    
    psf_rel_diff_1 = torch.linalg.norm(psf_finufft - psf_tkbn) / torch.linalg.norm(psf_finufft)
    print(f"Relative difference between Pytorch-Finufft and TkBN: {psf_rel_diff_1.item():.4e}")
    psf_rel_diff_2 = torch.linalg.norm(psf_finufft - psf_pynufft) / torch.linalg.norm(psf_finufft)
    print(f"Relative difference between Pytorch-Finufft and PyNUFFT: {psf_rel_diff_2.item():.4e}")
    psf_rel_diff_3 = torch.linalg.norm(psf_tkbn - psf_pynufft) / torch.linalg.norm(psf_tkbn)
    print(f"Relative difference between TkBN and PyNUFFT: {psf_rel_diff_3.item():.4e}")
