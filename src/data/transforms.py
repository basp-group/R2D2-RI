import numpy as np
import torch

def to_log(im, a=1000.0):
    """Logarithmic mapping of the image of interest according to the formula:
    rlog(x) = x_max * log(a * (x / x_max) + 1) / log(a),
    where a is the dynamic range.

    Parameters
    ----------
    im : torch.Tensor or np.ndarray
        Image of interest, of shape (B, C, H, W) for torch.Tensor or (H, W) for np.ndarray.
    a : int, optional
        _description_, by default 1000.0

    Returns
    -------
    _type_
        _description_
    """
    if 'Tensor' in str(type(im)):
        if 'tensor' not in str(type(a)):
            a_tensor = torch.tensor([a]).to(im.device)  # Convert 'a' to a tensor
        else:
            a_tensor = a.to(im.device)
        im_cur = torch.clamp(im, min=0)
        im_max = torch.amax(im_cur, dim=(-1, -2), keepdim=True)
        return im_max * torch.log10(a_tensor * (im_cur / im_max) + 1.) / torch.log10(a_tensor)
    else:
        im_cur = np.clip(im, 0, None)
        im_max = im_cur.max()
        return im_max * np.log10(a * (im_cur / im_max) + 1.) / np.log10(a)

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def normalize(data, mean, eps=0.):
    if type(data) != type(mean):
        if 'numpy' in str(type(data)) and 'torch' in str(type(mean)):
            mean = mean.numpy(force=True)
        # elif 'numpy' in str(type(data)) and 'cupy' in str(type(mean)):
        #     mean = mean.get()
        elif 'torch' in str(type(data)) and 'numpy' in str(type(mean)):
            mean = torch.from_numpy(mean)
        # elif 'torch' in str(type(data)) and 'cupy' in str(type(mean)):
        #     mean = torch.from_numpy(mean.get())
        # elif 'cupy' in str(type(data)) and 'numpy' in str(type(mean)):
        #     import cupy as cp
        #     mean = cp.asarray(mean)
        # elif 'cupy' in str(type(data)) and 'torch' in str(type(mean)):
        #     import cupy as cp
        #     mean = cp.asnumpy(mean)
    return data  / (mean + eps)

def normalize_instance(data, eps=0.):
    if 'numpy' in str(type(data)):
        mean = np.mean(data)
    # elif 'cupy' in str(type(data)):
    #     import cupy as cp
    #     mean = cp.mean(data)
        # mean = np.abs(mean) 
    elif 'torch' in str(type(data)):
        mean = data.mean(dim=(-1, -2), keepdim=True)
        # mean = torch.abs(mean)
    return normalize(data, mean, eps), mean

class DataTransform_N1:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):
        pass

    def __call__(self, target, dirty, PSF, fname, slice, a_expo):
        dirty = to_tensor(dirty.astype(np.float32))
        target = to_tensor(target.astype(np.float32))
        dirty_n, mean = normalize_instance(dirty, eps=1e-110)
        target_n = normalize(target, mean, eps=1e-110)
        a_expo = torch.tensor(np.array([a_expo]).astype(np.float32))
        if type(PSF) is not float:
            PSF = to_tensor(PSF.astype(np.float32))
        return dirty, dirty_n, target, target_n, PSF, fname, slice, mean, a_expo

class DataTransform_Ni:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):
        pass

    def __call__(self, target, res, rec, PSF, fname, slice, a_expo):
        rec = to_tensor(rec.astype(np.float32))
        res = to_tensor(res.astype(np.float32))
        target = to_tensor(target.astype(np.float32))

        rec_n, mean = normalize_instance(rec, eps=1e-110)
        target_n = normalize(target, mean, eps=1e-110)
        res_n = normalize(res, mean, eps=1e-110)
        a_expo = torch.tensor(np.array([a_expo]).astype(np.float32))
        if type(PSF) is not float:
            PSF = to_tensor(PSF.astype(np.float32))
        return res, res_n, target, target_n, rec, rec_n, PSF, fname, slice, mean, a_expo
