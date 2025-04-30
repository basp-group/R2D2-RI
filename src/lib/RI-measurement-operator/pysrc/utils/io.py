import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version

from .gen_imaging_weights import gen_imaging_weights
from .utils import vprint


def read_fits_as_tensor(path, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float64):
    x = fits.getdata(path)
    x = torch.tensor(x.astype(np.float32)).view(1, 1, *x.shape).to(device).to(dtype)
    return x


def load_data_to_tensor(
    uv_file_path: str,
    super_resolution: float = 1.5,
    image_pixel_size: float = None,
    data_weighting: bool = True,
    load_weight: bool = False,
    img_size: tuple = None,
    uv_unit: str = "radians",
    weight_type: str = "briggs",
    weight_gridsize: float = 2.0,
    weight_robustness: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    data: dict = None,
    verbose: bool = True,
):
    """Read u, v and imweight from specified path.

    Parameters
    ----------
    uv_file_path : str
        Path to the file containing sampling pattern, natural weights and (optional) imaging weights.
    super_resolution : float
        Super resolution factor.
    image_pixel_size : float, optional
        Image pixel size in arcsec, by default None
    data_weighting : bool, optional
        Flag to apply imaging weights, by default True
    load_weight : bool, optional
        Flag to load imaging weights from the file, by default False. If set to False and data_weighting is True, the imaging weights will be generated.
    weight_name : str, optional
        Name of the imaging weights in the data file, by default 'nWimag'
    dtype : torch.dtype, optional
        Data type to be used, by default torch.float64
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')
    verbose : bool, optional
        Flag to print information, by default True

    Returns
    -------
    data: dict
        Dictionary containing u, v, w, (optional) y, nW, (optional) nWimag and other information.
    """

    vprint(f"INFO: loading sampling pattern from {uv_file_path}", verbose)

    if data is None:
        data = {}

    # check mat file version and load data
    mat_version, _ = matfile_version(uv_file_path)
    if mat_version == 2:
        data = dict()
        with h5py.File(uv_file_path, "r") as h5File:
            for key, h5obj in h5File.items():
                if isinstance(h5obj, h5py.Dataset):
                    data[key] = np.array(h5obj)
                    if data[key].dtype.names and "imag" in data[key].dtype.names:
                        data[key] = data[key]["real"] + 1j * data[key]["imag"]
    else:
        data = loadmat(uv_file_path)

    u = data["u"].squeeze()
    v = data["v"].squeeze()
    if "w" in data:
        w = data["w"].squeeze()
    else:
        w = np.array([0.0])
    # convert uvw in units of the wavelength
    if "flag" in data:
        vprint("INFO: applying flagging to the sampling pattern", verbose)

        frequency = data["frequency"].squeeze()
        if len(frequency.shape) == 0:
            frequency = np.array([frequency.item()])

        flag = data["flag"]
        flag_counter = 0
        while len(flag.shape) > 3:
            flag = flag.squeeze(0)
            flag_counter += 1
            if flag_counter > 5:
                raise ValueError("Dimension of flags in the data cannot match dimension of the uv-points.")
        data["flag"] = torch.tensor(flag).to(device)

        nFreqs = data["nFreqs"].item()

        if "unit" in data and data["unit"].item() == "m":
            vprint("INFO: converting uv coordinate unit from meters to wavelength.", verbose)
            u = np.concatenate(
                [
                    u[flag[0, iFreq, :] == False] / (speed_of_light / frequency[iFreq].item())
                    for iFreq in range(nFreqs)
                ]
            )
            v = np.concatenate(
                [
                    v[flag[0, iFreq, :] == False] / (speed_of_light / frequency[iFreq].item())
                    for iFreq in range(nFreqs)
                ]
            )
            if "w" in data:
                w = np.concatenate([w[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
        else:
            u = np.concatenate([u[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
            v = np.concatenate([v[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
            if "w" in data:
                w = np.concatenate([w[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
    else:
        if "unit" in data:
            if data["unit"].item() == "m":
                vprint("INFO: converting uv coordinate unit from meters to wavelength.", verbose)
                wavelength = speed_of_light / data["frequency"].item()
                u = u / wavelength
                v = v / wavelength

    max_proj_baseline = np.max(np.sqrt(u**2 + v**2))
    data["max_proj_baseline"] = max_proj_baseline
    if "super_resolution" in data and data["super_resolution"].item() != super_resolution:
        vprint(
            f'INFO: super resolution factor in the data file is {data["super_resolution"].item():.4f}, while the user specified {super_resolution}.',
            verbose,
        )
        vprint(
            f'INFO: super resolution factor of {data["super_resolution"].item():.4f} will be used.', verbose
        )
        super_resolution = data["super_resolution"].item()
    elif "super_resolution" not in data:
        data["super_resolution"] = super_resolution
    if uv_unit == "radians":
        spatial_bandwidth = 2 * max_proj_baseline
        if image_pixel_size is not None:
            if verbose:
                vprint(f"INFO: user specified pixelsize: {image_pixel_size:.4e} arcsec.", verbose)
        else:
            if "nominal_pixelsize" in data:
                image_pixel_size = data["nominal_pixelsize"].item() / super_resolution
                if verbose:
                    vprint(
                        f"INFO: user-specified pixel size: {image_pixel_size:.4e} arcsec (i.e. super resolution factor: {super_resolution:.4f})",
                        verbose,
                    )
            else:
                image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
                if verbose:
                    vprint(
                        f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
                        verbose,
                    )

        data["image_pixel_size"] = image_pixel_size
        halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

        u = u * np.pi / halfSpatialBandwidth
        v = v * np.pi / halfSpatialBandwidth

    data["u"] = torch.tensor(u, dtype=dtype, device=device).view(1, 1, -1)
    data["v"] = -torch.tensor(v, dtype=dtype, device=device).view(1, 1, -1)
    data["w"] = -torch.tensor(w, dtype=dtype, device=device).view(1, 1, -1)

    if "nW" in data:
        if data["nW"].shape[-1] == 1 or data["nW"].shape[-1] == data["u"].shape[-1]:
            nW = data["nW"].squeeze()
        else:
            tau_index, nW_unique = zip(*sorted(zip(data["tau_index"].squeeze(), data["nW"].squeeze())))
            tau_index = tau_index + (max(data["u"].shape),)
            nW = np.zeros(max(data["u"].shape))
            for i in range(len(tau_index) - 1):
                nW[tau_index[i] : tau_index[i + 1]] = nW_unique[i]
        vprint("INFO: using provided nW.", verbose)
        data["nW"] = torch.tensor(nW, dtype=dtype, device=device).view(1, 1, -1)
    else:
        vprint(f'INFO: natural weights "nW" not found, set to 1.', verbose)
        data["nW"] = torch.tensor([1.0], dtype=dtype, device=device).view(1, 1, -1)

    if data_weighting:
        if load_weight:
            data["nWimag"] = data["nWimag"].squeeze()
            if data["nWimag"].size == 0:
                vprint("INFO: imaging weight is empty and will not be applied.", verbose)
                data["nWimag"] = [
                    1.0,
                ]
        else:
            vprint("INFO: computing imaging weights...", verbose)
            if "weight_robustness" in data:
                weight_robustness = data["weight_robustness"].item()
                vprint(f"INFO: load weight_robustness from data file {weight_robustness}", verbose)
            else:
                vprint(f"INFO: weight_robustness {weight_robustness}", verbose)
            data["nWimag"] = gen_imaging_weights(
                data["u"].clone(),
                data["v"].clone(),
                data["nW"],
                img_size,
                weight_type=weight_type,
                weight_gridsize=weight_gridsize,
                weight_robustness=weight_robustness,
            ).numpy(force=True)
    else:
        vprint("INFO: imaging weights will not be applied.", verbose)
        data["nWimag"] = [
            1.0,
        ]
    data["nWimag"] = torch.tensor(data["nWimag"], dtype=dtype, device=device).view(1, 1, -1)

    if "y" in data:
        if dtype == torch.float32:
            data["y"] = (
                torch.tensor(data["y"], device=device, dtype=torch.complex64).view(1, 1, -1)
                * data["nW"]
                * data["nWimag"]
            )
        else:
            data["y"] = (
                torch.tensor(data["y"], device=device, dtype=torch.complex128).view(1, 1, -1)
                * data["nW"]
                * data["nWimag"]
            )

    return data
