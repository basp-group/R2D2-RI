from pysrc.set_params import parse_args_meas_op
from pysrc.utils.io import load_data_to_tensor

from astropy.io import fits
import torch

def gen_meas_op(args, return_data: bool = False):
    # read Fourier sampling pattern from specified data file
    data = load_data_to_tensor(
        uv_file_path=args.data_file,
        super_resolution=args.super_resolution,
        image_pixel_size=args.image_pixel_size,
        data_weighting=args.data_weighting,
        load_weight=args.load_weight,
        img_size=args.img_size,
        dtype=args.meas_dtype,
        device=args.device,
        verbose=args.verbose,
    )

    # create measurement operator object based on the chosen nufft library
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
        grid_size=args.nufft_grid_size,
        kernel_dim=args.nufft_kernel_dim,
        mode=args.nufft_mode,
        device=args.device,
        dtype=args.meas_dtype,
    )

    # set the Fourier sampling pattern in the measurement operator
    if args.verbose:
        print("INFO: Setting Fourier sampling pattern in the measurement operator...")
    if return_data:
        return meas_op, data
    else:
        return meas_op


if __name__ == "__main__":
    args = parse_args_meas_op()
    meas_op = gen_meas_op(args)

    # compute normalised PSF
    dirac = torch.zeros(1, 1, *args.img_size, device=args.device, dtype=args.meas_dtype)
    dirac[..., args.img_size[0] // 2, args.img_size[1] // 2] = 1
    psf = meas_op.adjoint_op(meas_op.forward_op(dirac))
    ri_normalisation = psf.max()
    psf /= ri_normalisation
    
    fits.writeto("psf.fits", psf.squeeze().numpy(force=True), overwrite=True)