import os
import pathlib
from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
from torch.utils.data import Dataset

from .misc import vprint


def save_reconstructions(reconstructions, out_dir, rec_file_ext="_rec", res_file_ext="_dirty"):
    """Saves the reconstructions of the test dataset into fits files.

    Parameters
    ----------
    reconstructions : dict[str, np.array]
        A dictionary mapping input filenames to corresponding reconstructions.\
    out_dir : str
        Path to the output directory where the reconstructions should be saved.
    rec_file_ext : str, optional
        Extension for the reconstructed image file to be saved, by default '_rec'
    res_file_ext : str, optional
        Extension for the residual dirty image file, by default '_dirty'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split("/")[-1]
        fits.writeto(
            os.path.join(out_dir, fname.split(res_file_ext)[0], f"{rec_file_ext}.fits"),
            recons[0][1].squeeze().cpu().numpy(),
            overwrite=True,
        )


def read_fits_as_tensor(path):
    x = fits.getdata(path)
    x = torch.tensor(x.astype(np.float64)).view(1, 1, *x.shape)
    return x


def load_data_to_tensor(
    uv_file_path: str,
    super_resolution: float = 1.5,
    image_pixel_size: float = None,
    data_weighting: bool = True,
    load_time: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    data: dict = None,
    verbose: bool = True,
    uv_unit: str = "radians",
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
    load_die : bool, optional
        Flag to load DIEs from the file, by default False
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
    # check mat file version and load data
    vprint(f"INFO: loading uv file from {uv_file_path}", verbose)
    if data is None:
        data = {}
    mat_version, _ = matfile_version(uv_file_path)
    if mat_version == 2:
        with h5py.File(uv_file_path, "r") as h5File:
            for key, h5obj in h5File.items():
                if isinstance(h5obj, h5py.Dataset):
                    data[key] = np.array(h5obj)
                    if data[key].dtype.names and "imag" in data[key].dtype.names:
                        data[key] = data[key]["real"] + 1j * data[key]["imag"]
    else:
        loadmat(uv_file_path, mdict=data)
            
    try:
        data["uv_id"] = int(os.path.basename(uv_file_path).split("_id_")[-1].split("_")[0])
    except:
        data["uv_id"] = 1337

    if "super_resolution" in data and data["super_resolution"].item() != super_resolution:
        vprint(
            f'WARNING: super resolution in the uv file ({data["super_resolution"].item()}) does not match the provided super resolution ({super_resolution}).',
            verbose,
        )
        vprint(f"WARNING: overwriting specified super resolution with the one found in the uv file.", verbose)
        super_resolution = data["super_resolution"].item()
        data["super_resolution"] = data["super_resolution"].item()
    else:
        data["super_resolution"] = super_resolution

    # convert uvw in units of the wavelength
    u = data["u"].squeeze()
    v = data["v"].squeeze()
    if "flag" in data:
        vprint("INFO: applying flagging to the sampling pattern", verbose)
        frequency = data["frequency"].squeeze()
        if len(frequency.shape) == 0:
            frequency = np.array([frequency.item()])
        flag = data["flag"]
        if len(flag.shape) > 3:
            flag = flag.squeeze(0)
        elif len(flag.shape) == 2:
            flag = np.expand_dims(flag, axis=0)
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
        else:
            u = np.concatenate([u[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
            v = np.concatenate([v[flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
    else:
        if "unit" in data:
            if data["unit"].item() == "m":
                vprint("INFO: converting uv coordinate unit from meters to wavelength.", verbose)
                wavelength = speed_of_light / data["frequency"].item()
                u = u / wavelength
                v = v / wavelength

    max_proj_baseline = np.max(np.sqrt(u**2 + v**2))
    data["max_proj_baseline"] = max_proj_baseline
    spatial_bandwidth = 2 * max_proj_baseline
    if image_pixel_size is not None:
        vprint(f"INFO: user specified pixelsize: {image_pixel_size:.4e} arcsec,", verbose)
    else:
        if "nominal_pixelsize" in data:
            image_pixel_size = data["nominal_pixelsize"].item() / super_resolution

            vprint(
                f"INFO: user-specified pixel size: {image_pixel_size:.4e} arcsec (i.e. super resolution factor: {super_resolution:.4f})",
                verbose,
            )
        else:
            image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)

            vprint(
                f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
                verbose,
            )
    data["super_resolution"] = super_resolution

    data["u"] = torch.tensor(u, dtype=dtype, device=device).view(1, 1, -1)
    data["v"] = -torch.tensor(v, dtype=dtype, device=device).view(1, 1, -1)
    if uv_unit == "radians":
        halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

        data["u"] = data["u"] * np.pi / halfSpatialBandwidth
        data["v"] = data["v"] * np.pi / halfSpatialBandwidth

    c_dtype = torch.complex64 if dtype in [torch.float32, torch.float] else torch.complex128

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
        data["nW"] = torch.tensor(nW, dtype=c_dtype, device=device).view(1, 1, -1)
    elif verbose:
        vprint(f'INFO: natural weights "nW" not found, set to 1.', verbose)
        data["nW"] = torch.tensor([1.0], dtype=c_dtype, device=device).view(1, 1, -1)

    if data_weighting:
        vprint("INFO: imaging weights will be computed...", verbose)
    else:
        if verbose:
            vprint("INFO: imaging weights will not be applied.", verbose)
    data["nWimag"] = [
        1.0,
    ]

    data["nWimag"] = torch.tensor(data["nWimag"], dtype=c_dtype, device=device).view(1, 1, -1)

    if "y" in data:
        data["y"] = torch.tensor(data["y"], device=device, dtype=c_dtype).view(1, 1, -1)

    if load_time:
        time = np.concatenate([data["time"][flag[0, iFreq, :] == False] for iFreq in range(nFreqs)])
        data["time"] = torch.tensor(time, device=device)
    elif "time" in data:
        data.pop("time")

    if "timeStep" in data:
        data["timeStep"] = data["timeStep"].item()

    if "DR" in data:
        data["target_dynamic_range"] = data["DR"].squeeze()
    else:
        data["target_dynamic_range"] = None

    return data


class Data_N1(Dataset):
    """
    A PyTorch Dataset that provides access to image slices.
    """

    def __init__(self, hparams, data_partition, transform):
        self.transform = transform
        self.dirty = []
        self.clean = []
        self.mat_files = []
        self.hparams = hparams
        root = os.path.join(self.hparams.data_path, f"{data_partition}")  # ground truth
        root2 = os.path.join(
            self.hparams.data_path, f"{data_partition}{self.hparams.res_ext}"
        )  # dirty/ residual dirty
        mat_path = os.path.join(
            self.hparams.data_path, f"{data_partition}{self.hparams.mat_ext}"
        )  # dynamic range of the exponentiated image (1/a)
        if self.hparams.layers > 1:
            self.PSF = []
            PSF_path = self.hparams.PSF_path
        files_dt = list(pathlib.Path(root2).iterdir())

        for fname in sorted(files_dt):
            dirty_pth = str(fname.resolve())
            im = fits.getdata(dirty_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.dirty += [(dirty_pth, slice) for slice in range(num_slices)]

            clean_fname = (dirty_pth.split("/")[-1]).split(".fits")[0].split(self.hparams.dirty_file_ext)[0]
            if len(self.hparams.gdth_file_ext) > 0:
                clean_pth = os.path.join(root, f"{clean_fname}{self.hparams.gdth_file_ext}.fits")
            else:
                clean_pth = os.path.join(root, f"{clean_fname}.fits")
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]

            if self.hparams.layers > 1:
                uv_id = clean_fname.split("_id_")[-1]
                PSF_name = f"uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits"
                PSF_pth = os.path.join(PSF_path, PSF_name)
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]

            self.mat_files += [os.path.join(mat_path, f"{clean_fname}{self.hparams.mat_file_ext}.mat")]
        print("Number of dirty images:", len(self.dirty))
        print("Number of ground truth images:", len(self.clean))
        if self.hparams.layers > 1:
            print("Number of PSF images:", len(self.PSF))

    def __len__(self):
        return len(self.dirty)

    def __getitem__(self, i):
        fname, slice = self.clean[i]
        fname_dt, slice_dt = self.dirty[i]

        original_image = np.expand_dims(
            fits.getdata(fname), axis=0
        )  # Adding a channel dimension. shape of output is [1, W, H]
        dirty = np.expand_dims(
            fits.getdata(fname_dt), axis=0
        )  # Adding a channel dimension. shape of output is [1, W, H]

        if self.hparams.layers > 1:
            fname_PSF, slice_PSF = self.PSF[i]
            PSF = fits.getdata(fname_PSF)
            PSF = np.expand_dims(PSF / PSF.max(), axis=0)
        else:
            PSF = 0.0
        try:
            mat_file = loadmat(self.mat_files[i])
            try:
                a_expo = mat_file["a_expo"].squeeze()
            except:
                try:
                    a_expo = mat_file["expo_factor"].squeeze()
                except:
                    a_expo = 0.0
        except:
            a_expo = 0.0
        return self.transform(original_image, dirty, PSF, fname_dt.split("/")[-1], slice, a_expo)


class Data_Ni(Dataset):
    """
    A PyTorch Dataset that provides access to image slices.
    """

    def __init__(self, hparams, data_partition, transform):
        self.transform = transform
        self.res = []
        self.clean = []
        self.rec = []
        self.mat_files = []
        self.hparams = hparams
        root = os.path.join(self.hparams.data_path, f"{data_partition}")  # ground truth
        root2 = os.path.join(
            self.hparams.data_path, f"{data_partition}{self.hparams.res_ext}"
        )  # residual dirty
        root3 = os.path.join(
            self.hparams.data_path, f"{data_partition}{self.hparams.rec_ext}"
        )  # reconstruction
        mat_path = os.path.join(
            self.hparams.data_path, f"{data_partition}{self.hparams.mat_ext}"
        )  # dynamic range of the exponentiated image (1/a)
        if self.hparams.layers > 1:
            self.PSF = []
            PSF_path = self.hparams.PSF_path
        files_res = list(pathlib.Path(root2).iterdir())

        for fname in sorted(files_res):
            res_pth = str(fname.resolve())
            im = fits.getdata(res_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.res += [(res_pth, slice) for slice in range(num_slices)]

            clean_fname = (res_pth.split("/")[-1]).split("_res")[0]

            clean_pth = os.path.join(str(root.resolve()), clean_fname, f"{self.hparams.gdth_file_ext}.fits")
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]

            rec_pth = os.path.join(str(root3.resolve()), clean_fname, f"{self.hparams.rec_file_ext}.fits")
            im = fits.getdata(rec_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.rec += [(rec_pth, slice) for slice in range(num_slices)]

            if self.hparams.layers > 1:
                uv_id = clean_fname.split("_id_")[-1]
                PSF_name = f"uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits"
                PSF_pth = os.path.join(PSF_path, PSF_name)
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]

            self.mat_files += [os.path.join(mat_path, f"{clean_fname}{self.hparams.mat_file_ext}.mat")]

        print("rec path:", root3)
        print("Number of res images:", len(self.res))
        print("Number of clean images:", len(self.clean))

    def __len__(self):
        return len(self.res)

    def __getitem__(self, i):
        fname, slice = self.clean[i]
        fname_res, slice_res = self.res[i]
        fname_rec, slice_rec = self.rec[i]

        original_image = np.expand_dims(
            fits.getdata(fname), axis=0
        )  # Adding a channel dimension. shape of output is [1, W, H]
        res = np.expand_dims(
            fits.getdata(fname_res), axis=0
        )  # Adding a channel dimension. shape of output is [1, W, H]
        rec = np.expand_dims(fits.getdata(fname_rec), axis=0)

        if self.hparams.layers > 1:
            fname_PSF, slice_PSF = self.PSF[i]
            PSF = fits.getdata(fname_PSF)
            PSF = np.expand_dims(PSF / PSF.max(), axis=0)
        else:
            PSF = 0.0
        try:
            mat_file = loadmat(self.mat_files[i])
            try:
                a_expo = mat_file["a_expo"].squeeze()
            except:
                try:
                    a_expo = mat_file["expo_factor"].squeeze()
                except:
                    a_expo = 0.0
        except:
            a_expo = 0.0
        return self.transform(original_image, res, rec, PSF, fname_res.split("/")[-1], slice, a_expo)
