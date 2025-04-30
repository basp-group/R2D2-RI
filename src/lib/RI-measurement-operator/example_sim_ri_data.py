import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from scipy.io import savemat

from example_sim_measop import gen_meas_op
from pysrc.set_params import parse_args_sim_ri_data
from pysrc.utils.gen_imaging_weights import gen_imaging_weights
from pysrc.utils.io import read_fits_as_tensor

import timeit


def sim_ri_data(args):
    # Example script to simulate RI data
    # ground truth image
    gdthim = read_fits_as_tensor(args.gdth_file, args.device)
    # gdthim /= gdthim.max()
    plt.figure()
    # display the normalized ground truth image with peak pixel intensity = 1
    plt.imshow(gdthim.squeeze().numpy(force=True), cmap="afmhot")
    plt.colorbar()
    plt.title("Ground truth image")
    args.__setattr__("img_size", gdthim.shape[-2:])

    args.__setattr__(
        "nufft_grid_size",
        (
            int(args.img_size[0] * args.nufft_oversampling_factor),
            int(args.img_size[1] * args.nufft_oversampling_factor),
        ),
    )

    # create measurement operator object from specified arguments
    meas_op_raw, data = gen_meas_op(args, True)

    # model clean visibilities
    if args.verbose:
        print("Simulate model visibilities .. ")
    vis = meas_op_raw.forward_op(gdthim)
    # number of data points
    nmeas = max(vis.shape)

    if args.data_weighting and not args.load_weight:
        if args.verbose:
            print("Generate imaging weights...")
        if args.weight_type in ["uniform", "briggs"]:
            data["nWimag"] = gen_imaging_weights(
                data["u"].clone(),
                data["v"].clone(),
                data["nW"],
                args.img_size,
                weight_type=args.weight_type,
                weight_gridsize=args.weight_gridsize,
                weight_robustness=args.weight_robustness,
            )
        else:
            data["nWimag"] = 1.

    match args.nufft_pkg:
        case "finufft":
            from pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft

            Operator = MeasOpPytorchFinufft
        case "tkbn":
            from pysrc.measOperator.meas_op_nufft_tkbn import MeasOpTkbNUFFT

            Operator = MeasOpTkbNUFFT
        case "pynufft":
            from pysrc.measOperator.meas_op_nufft_pynufft import MeasOpPynufft

            Operator = MeasOpPynufft

    meas_op = Operator(
        u=data["u"],
        v=data["v"],
        img_size=args.img_size,
        real_flag=args.real_flag,
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        grid_size=args.nufft_grid_size,
        kernel_dim=args.nufft_kernel_dim,
        mode=args.nufft_mode,
        device=args.device,
        dtype=args.meas_dtype,
    )
    # noise vector
    if args.target_dynamic_range_heuristic:
        if args.verbose:
            print("Generate noise (noise level commensurate of the target dynamic range)...")
        # compute measop spectral norm to infer the noise heuristic
        # noise standard deviation heuristic
        # noise realization(mean-0; std-tau)
        t1 = timeit.default_timer()
        meas_op_norm = meas_op.get_op_norm(compute_flag=True, rel_tol=1e-4)
        print(f"Time to compute spectral norm: {timeit.default_timer() - t1}")
        if args.weight_type == "briggs":
            t2 = timeit.default_timer()
            meas_op_norm_prime = meas_op.get_op_norm_prime(rel_tol=1e-4)
            print(f"Time to compute spectral norm prime: {timeit.default_timer() - t2}")
            eta_correction = np.sqrt(meas_op_norm_prime / meas_op_norm)
        else:
            eta_correction = 1

        tau = np.sqrt(2 * meas_op_norm) / args.target_dynamic_range_heuristic / eta_correction

        # set random seed for reproducibility
        np.random.seed(1377)
        noise = tau * (torch.randn(nmeas) + 1j * torch.randn(nmeas)) / np.sqrt(2)
        noise = noise.to(args.device)
        # input signal to noise ratio
        isnr = 20 * torch.log10(torch.linalg.vector_norm(vis) / torch.linalg.vector_norm(noise))
        if args.verbose:
            print(f"INFO: random Gaussian noise with input SNR: {isnr:.3f} dB")
    elif args.noise_isnr:
        if args.verbose:
            print("generate noise from input SNR...")
        isnr = args.noise_isnr
        # user-specified input signal to noise ratio
        tau = np.linalg.norm(vis) / (10 ** (isnr / 20)) / np.sqrt((nmeas + 2 * np.sqrt(nmeas)))
        # set random seed for reproducibility
        np.random.seed(1377)
        noise = tau * (torch.randn(nmeas) + 1j * torch.randn(nmeas)) / np.sqrt(2)

    # data
    if args.verbose:
        print("Simulate data...")
    y = vis + noise

    # back-projected data
    data["nW"] = 1 / tau
    if args.verbose:
        print("Get back-projected data...")
    dirty = meas_op_raw.adjoint_op(y * data["nW"] ** 2 * data["nWimag"] ** 2)

    # display the non-normalized dirty image
    plt.figure()
    plt.imshow(dirty.squeeze().numpy(force=True), cmap="afmhot")
    plt.colorbar()
    extra_str = "(weights applied)"
    plt.title(f'(Non-normalised) dirty image {extra_str if args.data_weighting else ""}')
    if args.verbose:
        print("Done")
    y = y.numpy(force=True) if "torch" in str(type(y)) else y
    data_dict = {}
    data_dict.update(
        {
            "y": np.reshape(y, (nmeas, 1)),
            "nW": tau * np.ones((nmeas, 1)),
        }
    )
    if args.data_weighting:
        data_dict.update({"nWimag": np.reshape(data["nWimag"].numpy(force=True), (nmeas, 1))})

    savemat(os.path.join(args.result_path, f"data.mat"), data_dict)

    # save (non-normalized) dirty image
    fits.writeto(
        os.path.join(args.result_path, "dirty.fits"), dirty.numpy(force=True).squeeze(), overwrite=True
    )
    if args.verbose:
        print(f"Dirty image saved in {args.result_path}")


if __name__ == "__main__":
    args = parse_args_sim_ri_data()
    sim_ri_data(args)
