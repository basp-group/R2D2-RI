from typing import Literal, Optional, Tuple

import numpy as np
import torch


def gen_imaging_weights(u, v, nW, im_size, weight_type="briggs", weight_gridsize=2.0, weight_robustness=0.0):
    """
    Parameters
    ----------
    u : torch.tensor
        u coordinate of the data point in the Fourier domain in radians.
    v : torch.tensor
        v coordinate of the data points in the Fourier domain in radians.
    nW : torch.tensor
        square root of inverse of the variance.
    im_size:  tuple
        image dimension
    weight_type : str
        Type of weight: {'uniform', 'robust', 'none'}. Default is 'robust'.
    weight_gridsize : int
        Grid size for weight calculation. Default is 1.
    weight_robustness : float
        The Briggs parameter for robust weighting. Default is 0.0.

    Returns
    -------
    nWimag : torch.tensor
        weights inferred from the density of the sampling (uniform/Briggs).
    """
    
    u = u.clone()
    v = v.clone()

    nmeas = u.numel()
    N = torch.tensor([np.floor(i * weight_gridsize).astype(int) for i in im_size])
    nW2 = (nW.view(-1).to(u.device)) ** 2

    if nW2.size(-1) == 1:
        nW2 = nW2 * torch.ones(nmeas, device=u.device)
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]

    # Initialize gridded weights matrix with zeros
    p = torch.floor((v + torch.pi) * N[0] / 2 / torch.pi).to(torch.int64).view(-1) - 1
    q = torch.floor((u + torch.pi) * N[1] / 2 / torch.pi).to(torch.int64).view(-1) - 1
    gridded_weights = torch.zeros(torch.prod(N), dtype=torch.float64, device=u.device)
    uvInd = p * N[1] + q
    if weight_type != "none":
        if weight_type == "uniform":
            values = torch.ones_like(p, dtype=torch.float64)
        elif weight_type == "briggs":
            values = nW2

    # Use scatter_add_ to update gridded_weights
    gridded_weights.scatter_add_(0, uvInd, values.to(torch.float64))

    # Apply weighting based on weighting_type
    if weight_type != "none":
        gridded_vals = gridded_weights[uvInd]  # Gather the values

        if weight_type == "uniform":
            nWimag = 1 / torch.sqrt(gridded_vals)
        elif weight_type == "briggs":
            # Compute robust scale factor
            robust_scale = (torch.sum(nW2) / torch.sum(gridded_weights**2)) * (
                5 * 10 ** (-weight_robustness)
            ) ** 2
            nWimag = 1 / torch.sqrt(1 + robust_scale * gridded_vals)
    return nWimag.view(1, 1, -1)


def gen_imaging_weights_np(
    u: np.ndarray,
    v: np.ndarray,
    natural_weight: Optional[np.ndarray] = None,
    img_size: Tuple[int, int] = (512, 512),
    weight_type: Literal["uniform", "briggs", "robust"] = "uniform",
    weight_gridsize: int = 2,
    weight_robustness: float = 0.0,
):
    """
    Generate uniform weights or Briggs (robust) weights for RI measurements based on
    the uv sampling pattern.

    Args:
        u (np.ndarray): The u coordinates of the sampling pattern.
        v (np.ndarray): The v coordinates of the sampling pattern.
        img_size (Tuple[int, int]): The size of the image to be reconstructed.
        weight_type (Literal["uniform", "briggs", "robust"], optional):
            The type of weights to be generated. Can be "uniform" or "briggs".
            Defaults to "uniform".
        natural_weight (np.ndarray, optional): The natural weights for the uv sampling pattern.
            Defaults to None.
        grid_size (int, optional): The size of the grid for the uv sampling pattern.
            Defaults to 1.
        weight_robustness (float, optional): The robustness factor for the Briggs weighting.
            Defaults to 0.0.

    Returns:
        np.ndarray: The generated weights for the uv sampling pattern.

    Raises:
        NotImplementedError: If the weight_type is not "uniform" or "briggs".
    """
    # flatting u & v vector
    u = u.reshape((-1, 1)).astype(np.double)
    v = v.reshape((-1, 1)).astype(np.double)

    # consider only half of the plane
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]

    # Initialize
    nmeas = u.size
    weight_grid_size = np.floor(np.array((img_size[0] * weight_gridsize, img_size[1] * weight_gridsize)))
    gridded_weight = np.zeros(
        weight_grid_size.astype(int), dtype=np.double
    )  # Initialize gridded weights matrix with zeros
    image_weight = np.ones((nmeas, 1))

    # grid uv points
    q = np.floor((u + np.pi) * weight_grid_size[1] / 2.0 / np.pi).astype(int) - 1
    p = np.floor((v + np.pi) * weight_grid_size[0] / 2.0 / np.pi).astype(int) - 1  # matching index in matlab

    if weight_type == "uniform":
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += 1.0
        # Apply weighting
        image_weight = 1.0 / np.sqrt(gridded_weight[p, q])
    elif weight_type in ["robust", "briggs"]:
        # inverse of the noise variance
        natural_weight = natural_weight.reshape((-1, 1)).astype(np.double)
        natural_weight2 = natural_weight**2
        if natural_weight2.size == 1:
            natural_weight2 = natural_weight2[0] * np.ones((nmeas, 1))
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += natural_weight2[idx]
        # Compute robust scale factor
        robust_scale = (np.sum(gridded_weight) / np.sum(gridded_weight**2)) * (
            5 * 10 ** (-weight_robustness)
        ) ** 2
        # Apply weighting
        image_weight = 1.0 / np.sqrt(1.0 + robust_scale * gridded_weight[p, q])
    else:
        raise NotImplementedError("Image weighting type: " + weight_type)

    return image_weight
