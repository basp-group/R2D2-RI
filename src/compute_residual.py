"""
Compute residual dirty images for the given reconstructed images and dirty images, with measurement operator constructed from sampling pattern.
"""

import os
import pathlib

import numpy as np
import torch
from astropy.io import fits
from scipy.io import loadmat
from tqdm import tqdm

from lib import gen_imaging_weights
from utils import (
    create_meas_op,
    parse_args_compute_residual,
    read_fits_as_tensor,
    load_data_to_tensor,
    vprint,
)

@torch.no_grad()
def compute_residual(args):
    """Compute and save residual dirty images for the given reconstructed imagesm dirty images and uv data.

    Parameters
    ----------
    uv_path : str
        Path to uv data.
    rec_path : str
        Path to reconstructed images.
    dirty_path : str
        Path to dirty images.
    output_res_path : str
        Path to save the residual dirty images.
    super_resolution : float, optional
        Super resolution factor, by default 1.
    prune : bool, optional
        If True, dataset will be pruned, by default False.
    imweight_name : str, optional
        Specific name of variable containing the weighting to be used in the uv file, by default 'nWimag'.
    on_gpu : bool, optional
        If True, dirty images will be computed on gpu, by default False.
    """
    rec_paths = list(pathlib.Path(args.rec_path).iterdir())
    img_size = fits.getdata(rec_paths[0]).squeeze().shape
    
    if args.img_size != img_size:
        args.img_size = img_size
        nufft_grid_size = tuple(int(i * args.nufft_oversampling_factor) for i in args.img_size)
        args.nufft_grid_size = nufft_grid_size

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    output_res_path = os.path.join(
        args.output_path, f"{args.dataset}set_fullnumpy{args.output_res_path_suffix}"
    )
    if not os.path.exists(output_res_path):
        os.makedirs(output_res_path, exist_ok=True)
    if args.prune:
        high_path = os.path.join(output_res_path, "high")
        low_path = os.path.join(output_res_path, "low")
        os.makedirs(high_path, exist_ok=True)
        os.makedirs(low_path, exist_ok=True)

    for rec_file in tqdm(rec_paths):
        fname = rec_file.name.split("_rec")[0]
        mat_file = os.path.join(args.mat_path, f"{fname}.mat")
        data = {}
        loadmat(mat_file, data, variable_names=["nW", "tau_index", "weight_robustness"])

        if (
            os.path.exists(os.path.join(output_res_path, f"{fname}_res.fits"))
            or (args.prune and os.path.exists(os.path.join(high_path, f"{fname}_res.fits")))
            or (args.prune and os.path.exists(os.path.join(low_path, f"{fname}_res.fits")))
        ):
            continue
        else:
            rec = read_fits_as_tensor(rec_file).to(args.device)
            fname_uv = fname.split("_id")[1].split("_new_peak")[0]
            uv_file = os.path.join(args.uv_path, f"uv_id{fname_uv}.mat")
            data = load_data_to_tensor(
                uv_file_path=uv_file,
                super_resolution=args.super_resolution,
                image_pixel_size=args.image_pixel_size,
                data_weighting=args.data_weighting,
                load_die=args.load_die,
                dtype=args.meas_dtype,
                device=args.device,
                data=data,
                verbose=args.verbose,
            )
            if args.data_weighting:
                vprint("Generating imaging weights ...", args.verbose)
                vprint(f"INFO: weight type: {args.weight_type}", args.verbose)
                if args.weight_type == "briggs" and "weight_robustness" in data:
                    weight_robustness = data["weight_robustness"].item()
                    if weight_robustness != args.weight_robustness:
                        vprint(
                            f"INFO: weight robustness found in data file, using {weight_robustness}.",
                            args.verbose,
                        )
                elif args.weight_type == "briggs":
                    weight_robustness = args.weight_robustness
                    vprint(f"INFO: weight robustness: {weight_robustness}", args.verbose)
                else:
                    weight_robustness = None
                
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

            dirty_file = os.path.join(args.dirty_path, f'{fname.split("/")[-1]}_dirty.fits')
            dirty = read_fits_as_tensor(dirty_file).to(args.device)

            psf = meas_op.get_psf()
            psf_peak = psf.max()
            y = meas_op.forward_op(rec)
            if args.load_die:
                y = y * data["die"]
            res = dirty - meas_op.adjoint_op(y) / psf_peak
            res = res.numpy(force=True).squeeze()
            if args.prune:
                res_norm = np.linalg.norm(res.flatten())
                assert args.mat_path is not None, "mat_path should be provided for pruning."

                loadmat(mat_file, data, variable_names=["true_noise_norm"])
                epsilon = data["true_noise_norm"].item()
                res_norm_sqr = res_norm**2
                if epsilon < res_norm_sqr:
                    fits.writeto(
                        os.path.join(output_res_path, "high", f"{fname}_res.fits"),
                        res,
                        overwrite=True,
                    )
                else:
                    fits.writeto(
                        os.path.join(output_res_path, "low", f"{fname}_res.fits"),
                        res,
                        overwrite=True,
                    )
            else:
                fits.writeto(os.path.join(output_res_path, f"{fname}_res.fits"), res, overwrite=True)


if __name__ == "__main__":
    args = parse_args_compute_residual()
    compute_residual(args)
