"""
RI data simulation to create visibilities (complex measurement) and (optional) dirty imgage, PSF with provided sampling pattern and measurement operator parameters.
"""

import os
import pathlib

import numpy as np
import torch
from astropy.io import fits
from scipy.io import loadmat, savemat
from tqdm import tqdm

from lib import gen_imaging_weights
from utils import (
    create_meas_op,
    parse_args_ri_data_simulation,
    compute_tau,
    expo_im,
    solve_expo_factor,
    read_fits_as_tensor,
    load_data_to_tensor,
    vprint,
)

@torch.no_grad()
def simulate(args):
    """Generate and save unweighted visibilities for the given ground truth images and uv data.

    Parameters
    ----------
    uv_path : str
        Path to uv data.
    gdth_path : str
        Path to the ground truth images.
    output_dirty_path : str
        Path to save the dirty images.
    output_gdth_path : str
        Path to save the exponentiated gdth images.
    super_resolution : float, optional
        Super resolution factor, by default 1.
    sigma_range : tuple, optional
        If a tuple of (min, max) is provided, sigma will be uniformly sampled from
        U[min, max] and corresponding noise will be added to the measurement, by default None.
    briggs : bool, optional
        If True, briggs weighting will be applied, by default False.
    expo : bool, optional
        If True, ground truth will be exponentiated, by default False.
    sigma0 : float, optional
        1/ current dynamic range of the ground truth image, by default 0.
    imweight_name : str, optional
        Specific name of variable containing the weighting to be used in the uv file, by default 'nWimag'.
    on_gpu : bool, optional
        If True, dirty images will be computed on gpu, by default False.
    """
    vprint(f"Generating dirty images for {args.dataset} set ...", args.verbose)
    gdth_paths = [i for i in list(pathlib.Path(args.gdth_path).iterdir()) if str(i).endswith("fits")]
    vprint(f"from {len(gdth_paths)} ground truth images ...", args.verbose)

    # Create main output path and necessary subdirectories
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    output_dirty_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_dirty")
    if not os.path.exists(output_dirty_path) and args.save_dirty:
        os.makedirs(output_dirty_path, exist_ok=True)
        vprint(f"Created output dirty path: {output_dirty_path}", args.verbose)

    output_PSF_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_PSF")
    if not os.path.exists(output_PSF_path) and args.save_PSF:
        os.makedirs(output_PSF_path, exist_ok=True)
        vprint(f"Created output PSF path: {output_PSF_path}", args.verbose)

    output_data_path = os.path.join(args.output_path, f"{args.dataset}set_data")
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path, exist_ok=True)
        vprint(f"Created output data path: {output_data_path}", args.verbose)

    # Exponentiated ground truth images will be saved if expo is True
    if args.expo:
        output_gdth_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_gdth")
        if not os.path.exists(output_gdth_path):
            vprint(f"Created output ground truth path: {output_data_path}", args.verbose)
            os.makedirs(output_gdth_path, exist_ok=True)

    if type(args.seed) == int:
        seed = args.seed

    if args.uv_random:
        uv_files = list(pathlib.Path(args.uv_path).iterdir())
        uv_files = [uv for uv in uv_files if uv.suffix == ".mat"]
        uv_files = np.random.permutation(uv_files)

    pbar = tqdm(gdth_paths)
    for idx, gdth_file in enumerate(pbar):
        gdth = read_fits_as_tensor(gdth_file).to(args.device)
        img_size = gdth.squeeze().shape
        vprint(f"Working on {gdth_file.name} ...", args.verbose)
        fname = gdth_file.name.split(".fits")[0].split("_gdth")[0]
        if args.uv_random:
            uv_file = uv_files[idx]
            uv_fname = uv_file.name.split(".mat")[0].split("uv_id_")[-1]
            fname_uv = uv_fname
            fname = f"{fname}_id_{uv_fname}"
        else:
            fname_uv = fname.split("_id_")[1]
            uv_file = pathlib.Path(os.path.join(args.uv_path, f"uv_id_{fname_uv}.mat"))
            uv_fname = uv_file.name.split(".mat")[0].split("uv_id_")[-1]

        if args.seed == "uv":
            uv_id = int(fname_uv.split("_")[0])
            np.random.seed(uv_id)
        else:
            np.random.seed(seed + idx)

        mat_file_path = os.path.join(output_data_path, f"{fname}.mat")
        gdth_path = os.path.join(output_gdth_path, f"{fname}_gdth.fits")
        dirty_path = os.path.join(output_dirty_path, f"{fname}_dirty.fits")

        if os.path.exists(mat_file_path) and os.path.exists(dirty_path) and os.path.exists(gdth_path):
            vprint(f"Data already exists for {fname}, skipping ...", args.verbose)
            continue

        data = load_data_to_tensor(
            uv_file_path=uv_file,
            super_resolution=args.super_resolution,
            image_pixel_size=args.image_pixel_size,
            data_weighting=args.data_weighting,
            load_die=args.load_die,
            load_time=args.multi_noise,
            dtype=args.meas_dtype,
            device=args.device,
            verbose=args.verbose,
        )

        args.img_size = img_size
        nufft_grid_size = tuple(int(i * args.nufft_oversampling_factor) for i in img_size)
        args.nufft_grid_size = nufft_grid_size

        meas_op_raw = create_meas_op(args, data, os=1, device=args.device)

        tau, sigma = compute_tau(meas_op_raw, data, args.sigma_range_min, args.sigma_range_max, args.device)

        data["nW"] = 1 / tau.unsqueeze(1) if max(tau.shape) > 1 else 1 / tau

        mat_dict = {"sigma": sigma.squeeze().numpy(force=True)}
        mat_dict.update({"super_resolution": data["super_resolution"]})

        if args.data_weighting:
            vprint("Generating imaging weights ...", args.verbose)
            vprint(f"INFO: weight type: {args.weight_type}", args.verbose)
            if args.weight_type == "briggs":
                if "weight_robustness" not in data:
                    weight_robustness = 0.0
                else:
                    weight_robustness = data["weight_robustness"].item()
                vprint(f"INFO: weight robustness: {weight_robustness}", args.verbose)
                mat_dict.update({"weight_robustness": weight_robustness})
            else:
                weight_robustness = 0.0

            data["nWimag"] = (
                gen_imaging_weights(
                    data["u"].clone(),
                    data["v"].clone(),
                    data["nW"],
                    img_size,
                    args.weight_type,
                    args.weight_gridsize,
                    torch.tensor(weight_robustness).to(args.device),
                )
                .to(args.device)
                .view(1, 1, -1)
            )

        meas_op = create_meas_op(args, data, os=1, device=args.device)

        if args.data_weighting:
            vprint("INFO: Computing eta_briggs ...", args.verbose)
            meas_op_norm = meas_op.get_op_norm()
            meas_op_norm_prime = meas_op.get_op_norm_prime()
            eta_briggs = np.sqrt(meas_op_norm_prime / meas_op_norm)
            vprint(f"INFO: eta_briggs: {eta_briggs.item():.4f}", args.verbose)
            alpha = sigma * np.sqrt(2 * meas_op_norm)
            vprint(f"INFO: correction factor alpha: {alpha}", args.verbose)
            tau = tau * alpha / eta_briggs

        # Compute exponentiation factor and save exponentiated ground truth images
        if args.expo:
            assert sigma < args.sigma0, "sigma should be greater than sigma0 for exponentiation."
            expo_factor = solve_expo_factor(args.sigma0, sigma.numpy(force=True))
            vprint(
                f"Ground truth exponentiated with factor: {expo_factor} to target dynamic range {1/sigma.item():.4f}",
                args.verbose,
            )
            gdth = expo_im(gdth, expo_factor)
            fits.writeto(
                gdth_path,
                gdth.squeeze().numpy(force=True),
                overwrite=True,
            )

        # Generate unweighted visibilities
        y = meas_op_raw.forward_op(gdth)
        if args.load_die:
            y = y * data["die"]
        noise_y = (torch.randn_like(y) + 1j * torch.randn_like(y)) * tau / np.sqrt(2)
        y += noise_y
        if args.save_vis:
            # save auxiliary data related to sampling pattern, weighting and noise
            uv_data = loadmat(uv_file, variable_names=["u", "v", "flag", "unit", "frequency", "nFreqs"])
            u = uv_data["u"]
            v = uv_data["v"]

            mat_dict.update(
                {
                    "u": u,
                    "v": v,
                    "frequency": uv_data["frequency"],
                    "nFreqs": uv_data["nFreqs"],
                    "y": y.squeeze().unsqueeze(1).numpy(force=True),
                }
            )

            if "unit" in uv_data and uv_data["unit"].item() == "m":
                mat_dict.update({"unit": "m"})

            if "flag" in uv_data:
                mat_dict.update({"flag": uv_data["flag"]})

        data["nW"] = 1 / tau
        data["nW"] = torch.tensor(data["nW"]).view(1, 1, -1).to(args.device)
        meas_op = create_meas_op(args, data, os=1, device=args.device)

        psf = meas_op.get_psf()
        psf_peak = psf.max()
        noise = meas_op.adjoint_op(noise_y * data["nWimag"] * data["nW"]) / psf_peak
        true_noise_norm = np.linalg.norm(noise.squeeze().numpy(force=True)) ** 2
        mat_dict.update({"noise": noise.squeeze().numpy(force=True), "true_noise_norm": true_noise_norm})
        # Compute dirty image
        if args.save_dirty:
            dirty = meas_op.adjoint_op(y * data["nWimag"] * data["nW"]) / psf_peak
            fits.writeto(
                dirty_path,
                dirty.squeeze().numpy(force=True),
                overwrite=True,
            )

        # Compute PSF
        if args.save_PSF:
            fits.writeto(
                os.path.join(output_PSF_path, f"{fname}_PSF.fits"),
                psf.squeeze().numpy(force=True),
                overwrite=True,
            )

        if max(tau.shape) > 1:
            tau = tau.squeeze().numpy(force=True)
        else:
            tau = tau.item()

        tau_unique, tau_index = np.unique(tau, return_index=True)
        mat_dict.update(
            {
                "tau": tau_unique,
                "nW": 1 / tau_unique,
                "tau_index": tau_index,
                "maxProjBaseline": data["max_proj_baseline"].item(),
            }
        )

        if args.expo:
            mat_dict.update({"expo_factor": expo_factor})
            mat_dict.update({"DR": 1 / sigma.item()})

        savemat(mat_file_path, mat_dict)
        vprint(f"Data saved to {mat_file_path}", args.verbose)
        vprint("-" * 50, args.verbose)


if __name__ == "__main__":
    args = parse_args_ri_data_simulation()
    simulate(args)
