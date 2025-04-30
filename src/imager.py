import argparse
import os

import torch
import numpy as np
from astropy.io import fits

from lib import gen_imaging_weights
from optimiser import R2D2
from utils import load_data_to_tensor, parse_args_imaging, vprint, create_meas_op, snr, to_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2D2 image reconstruction.")
    args = parse_args_imaging()
    data = load_data_to_tensor(
        uv_file_path=args.data_file,
        super_resolution=args.super_resolution,
        image_pixel_size=args.image_pixel_size,
        data_weighting=True,
        dtype=args.meas_dtype,
        device=args.device,
        verbose=args.verbose,
    )

    vprint("Generating imaging weights ...", args.verbose)
    vprint(f"INFO: weight type: {args.weight_type}", args.verbose)
    if args.weight_type == "briggs":
        if "weight_robustness" in data:
            weight_robustness = data["weight_robustness"].item()
            vprint(f"INFO: weight robustness found in data file.", args.verbose)
        else:
            weight_robustness = args.weight_robustness
        vprint(f"INFO: weight robustness: {weight_robustness}", args.verbose)
    else:
        weight_robustness = None
    vprint(f"INFO: computing imaging weights ...", args.verbose)
    data["nWimag"] = (
        gen_imaging_weights(
            data["u"].clone(),
            data["v"].clone(),
            data["nW"],
            args.img_size,
            args.weight_type,
            args.weight_gridsize,
            torch.tensor(weight_robustness).to(args.device),
        )
        .to(args.device)
        .view(1, 1, -1)
    ).to(torch.complex128 if args.meas_dtype == torch.float64 else torch.complex64)

    meas_op = create_meas_op(args=args, data=data, device=args.device)

    if args.target_dynamic_range is None and "sigma" not in data:
        target_dynamic_range = np.sqrt(2 * meas_op.get_op_norm())
        vprint(
            f"Estimating target dynamic range as the reciprocal of the heuristic 1/sqrt(2L) = {target_dynamic_range:4e}",
            args.verbose,
        )
    elif args.target_dynamic_range is None:
        target_dynamic_range = 1 / data["sigma"].item() if "sigma" in data else args.target_dynamic_range
    else:
        target_dynamic_range = args.target_dynamic_range

    if not args.src_name:
        args.src_name = os.path.basename(args.data_file).split(".mat")[0]
    save_pth_main = os.path.join(args.output_path, args.src_name)
    os.makedirs(save_pth_main, exist_ok=True)

    if args.layers > 1 or args.data_init:
        meas_op2 = create_meas_op(args=args, data=data, os=2, device=args.device)
    else:
        meas_op2 = None

    if args.ckpt_realisations == 1:
        optimiser = R2D2(
            meas=data["y"] * data["nW"] * data["nWimag"],
            meas_op=meas_op,
            ckpt_path=args.ckpt_path,
            layers=args.layers,
            num_iter=args.num_iter,
            meas_op2=meas_op2,
            save_pth=save_pth_main,
            save_all_outputs=args.save_all_outputs,
            gdth_file=args.gdth_file,
            target_dynamic_range=target_dynamic_range,
            architecture=args.architecture,
            num_chans=args.num_chans,
            prune=args.prune,
            sigma_res_tol=args.sigma_res_tol,
            input_order=args.input_order,
            device=args.device,
            verbose=args.verbose,
        )

        optimiser.initialisation()
        optimiser.run()
        optimiser.finalisation()

    elif args.ckpt_realisations > 1:
        for i in range(args.ckpt_realisations):
            assert os.path.exists(
                os.path.join(args.ckpt_path, f"V{i+1}")
            ), f"Checkpoint path not found: {os.path.join(args.ckpt_path, f'V{i+1}')}"

        vprint(
            f"{args.ckpt_realisations} realisations of R2D2 series specified, epistemic uncertainty quantification will also be computed.",
            args.verbose,
        )
        for i in range(args.ckpt_realisations):
            vprint("", args.verbose)
            vprint("=" * 50, args.verbose)
            vprint(f"Realisation {i+1}/{args.ckpt_realisations}:", args.verbose)
            save_pth = os.path.join(save_pth_main, f"V{i+1}")
            os.makedirs(save_pth, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_path, f"V{i+1}")
            optimiser = R2D2(
                meas=data["y"] * data["nW"] * data["nWimag"],
                meas_op=meas_op,
                ckpt_path=ckpt_path,
                layers=args.layers,
                num_iter=args.num_iter,
                meas_op2=meas_op2,
                save_pth=save_pth,
                save_all_outputs=args.save_all_outputs,
                gdth_file=args.gdth_file,
                target_dynamic_range=target_dynamic_range,
                architecture=args.architecture,
                num_chans=args.num_chans,
                prune=args.prune,
                sigma_res_tol=args.sigma_res_tol,
                input_order=args.input_order,
                device=args.device,
                verbose=args.verbose,
            )

            optimiser.initialisation()
            optimiser.run()
            optimiser.finalisation()

        rec = np.zeros((args.ckpt_realisations, *args.img_size))
        for i in range(args.ckpt_realisations):
            rec[i] = fits.getdata(os.path.join(save_pth_main, f"V{i+1}", "R2D2_model_image.fits"))
        rec_mean = rec.mean(axis=0)
        rec_std = rec.std(axis=0)

        mask = rec_mean > (1 / target_dynamic_range)
        std_over_avg = np.zeros(args.img_size)
        std_over_avg[mask] = rec_std[mask] / rec_mean[mask]
        MRU = std_over_avg.mean()
        std_over_avg[std_over_avg == 0] = np.nan
        fits.writeto(
            os.path.join(save_pth_main, "R2D2_std_over_mean_image.fits"),
            std_over_avg.squeeze(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(save_pth_main, "R2D2_mean_model_image.fits"),
            rec_mean.squeeze(),
            overwrite=True,
        )
        if args.gdth_file is not None:
            gdth = fits.getdata(args.gdth_file)
            snr_tmp = snr(gdth, rec_mean, 1e-110)
            vprint(f"Computing metrics with the model image averaged over the {args.ckpt_realisations} realisations.", args.verbose)
            vprint(f"SNR: {snr_tmp:.4f} dB", args.verbose)
            if target_dynamic_range is not None:
                log_snr_tmp = snr(to_log(gdth, target_dynamic_range), to_log(rec_mean, target_dynamic_range), 1e-110)
                vprint(f"logSNR: {log_snr_tmp:.4f} dB", args.verbose)
        vprint(f"MRU: {MRU:.4e}", args.verbose)