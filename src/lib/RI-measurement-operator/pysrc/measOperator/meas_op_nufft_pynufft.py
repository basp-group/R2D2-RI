"""
Measurement operators built on NUFFT using PyNUFFT library
"""

import gc
from typing import Tuple

import numpy as np
import scipy
import torch
from pynufft.src._helper.helper import plan

from .meas_op_nufft import MeasOpNUFFT


class MeasOpPynufft(MeasOpNUFFT):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Use PyNUFFT as the NUFFT package, which translates Fessler's Michigan
    Image Reconstruction Toolbox (MIRT) NUFFT from MATLAB to Python, uses
    sparse G matrix and Kaiser-Bessel kernel.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        img_size: Tuple[int, int],
        natural_weight: torch.Tensor = torch.ones(1, 1),
        image_weight: torch.Tensor = torch.ones(1, 1),
        precond_weight: torch.Tensor = torch.ones(1, 1),
        real_flag: bool = True,
        grid_size: Tuple[int, int] = None,
        kernel_dim: int = 7,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        cal_time: bool = False,
        sparse_format: str = "csr",
        verbose: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the measurement operator.

        :param u: The u coordinates of the sampling points in units of radians.
        :type u: torch.Tensor
        :param v: The v coordinates of the sampling points in units of radians.
        :type v: torch.Tensor
        :param img_size: The target image size.
        :type img_size: Tuple[int, int]
        :param natural_weight: The natural weight for the measurement operator, defaults to torch.ones(1, 1)
        :type natural_weight: torch.Tensor, optional
        :param image_weight: The image weight for the measurement operator, defaults to torch.ones(1, 1)
        :type image_weight: torch.Tensor, optional
        :param precond_weight:  The preconditioning weight for the measurement operator, defaults to torch.ones(1, 1)
        :type precond_weight: torch.Tensor, optional
        :param real_flag: Flag to indicate the image is real of complex, defaults to True
        :type real_flag: bool, optional
        :param grid_size: The size of the oversampled Fourier grid, defaults to None
        :type grid_size: Tuple[int, int], optional
        :param kernel_dim: The dimension of the Kaiser-Bessel kernel, defaults to 7
        :type kernel_dim: int, optional
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
        :param cal_time: Enable to compute each forward and adjoint operator timing, defaults to False
        :type cal_time: bool, optional
        :param sparse_format: The format of the sparse G matrix, defaults to "csr"
        :type sparse_format: str, optional
        :param verbose: Enable the verbose output in w-projection, defaults to True
        :type verbose: bool, optional
        """
        super().__init__(
            u=u,
            v=v,
            img_size=img_size,
            natural_weight=natural_weight,
            image_weight=image_weight,
            precond_weight=precond_weight,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
            cal_time=cal_time,
        )

        self._traj = self._traj.squeeze(0)
        if grid_size is None:
            self._grid_size = tuple(np.array(self._img_size) * 2)
        else:
            self._grid_size = grid_size
        self._kernel_dim = (kernel_dim, kernel_dim)
        self._img_size = tuple(img_size)
        self._sparse_format = sparse_format
        self._verbose = verbose

        self._compute_G()

    def _compute_G(self):
        """
        Compute the sparse G matrix, apply w-projection if w-terms are provided and if necessary, mask the G matrix and move it to the device as a sparse tensor.
        """

        @self.conditional_decorator(self.time_decorator, self._cal_time)
        def compute_G_inner():
            st = plan(
                om=self._traj.T.numpy(force=True),
                Nd=self._img_size,
                Kd=self._grid_size,
                Jd=self._kernel_dim,
                format="CSR",
            )
            G = st["p"]
            self.mask_G = np.array((np.sum(np.abs(G), axis=0))).squeeze().astype("bool")
            G = G[:, self.mask_G]
            G_size = G.shape

            # convert the sparse G scipy csr matrix to Pytorch sparse tensor
            if self._sparse_format == "coo":
                self.G_sp = self.sparse_mx_to_torch_sparse_tensor(
                    G, G_size, dtype=self._dtype_meas, device=self._device
                )
            elif self._sparse_format == "csr":
                self.G_sp = self.csr_matrix_to_torch_csr_tensor(
                    G, G_size, dtype=self._dtype_meas, device=self._device
                )
            self.sn = torch.tensor(st["sn"], dtype=self._dtype, device=self._device)
            del G, st
            gc.collect()

        compute_G_inner()

    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight

        :param x: The input tensor in the image domain.
        :type x: torch.Tensor
        :return: The result of the forward NUFFT operation.
        :rtype: torch.Tensor
        """

        xx = self.sn * x
        k = torch.zeros(self._grid_size, dtype=self._dtype_meas).to(self._device)
        k[list(slice(self._img_size[jj]) for jj in range(len(self._img_size)))] = xx
        k = torch.fft.fftn(k).view(-1)[self.mask_G]
        return self.G_sp.mv(k).view(1, 1, -1) * self._data_weight

    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint NUFFT with data weight.

        :param y: The input tensor in the Fourier domain.
        :type y: torch.Tensor
        :return: The result of the adjoint NUFFT operation.
        :rtype: torch.Tensor
        """
        k_tmp = torch.zeros(np.prod(self._grid_size), dtype=self._dtype_meas).to(self._device)
        k_tmp[self.mask_G] = self.G_sp.H.mv((y * self._data_weight).view(-1))
        k = torch.reshape(k_tmp, self._grid_size)
        xx = torch.fft.ifftn(k, norm="forward")[
            list(slice(self._img_size[jj]) for jj in range(len(self._img_size)))
        ]

        return (xx * self.sn).view(1, 1, *self._img_size)

    @torch.no_grad()
    def sparse_mx_to_torch_sparse_tensor(
        self,
        sparse_mx: scipy.sparse.csr_matrix,
        sparse_mx_shape: tuple[int, int],
        dtype=torch.complex128,
        device=torch.device("cpu"),
    ) -> torch.sparse_coo_tensor:
        """
        Convert a scipy sparse matrix to a torch sparse COO tensor.

        https://github.com/DSE-MSU/DeepRobust
        :param sparse_mx: The scipy sparse matrix.
        :type sparse_mx: scipy.sparse.csr_matrix
        :param sparse_mx_shape: The shape of the sparse matrix.
        :type sparse_mx_shape: Tuple[int, int]
        :param dtype: The data type of the sparse tensor, defaults to torch.complex128
        :type dtype: torch.dtype, optional
        :param device: The device of the sparse tensor, defaults to torch.device("cpu")
        :type device: torch.device, optional
        """
        sparse_mx = sparse_mx.tocoo().astype(np.complex64)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx_shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)

    @torch.no_grad()
    def csr_matrix_to_torch_csr_tensor(
        self,
        sparse_mx: scipy.sparse.csr_matrix,
        sparse_mx_shape: tuple[int, int],
        dtype=torch.complex128,
        device=torch.device("cpu"),
    ) -> torch.sparse_csr_tensor:
        """
        Convert a scipy sparse matrix to a torch sparse tensor.
        :param sparse_mx: The scipy sparse matrix.
        :type sparse_mx: scipy.sparse.csr_matrix
        :param sparse_mx_shape: The shape of the sparse matrix.
        :type sparse_mx_shape: Tuple[int, int]
        :param dtype: The data type of the sparse tensor, defaults to torch.complex128
        :type dtype: torch.dtype, optional
        :param device: The device of the sparse tensor, defaults to torch.device("cpu")
        :type device: torch.device, optional
        """
        return torch.sparse_csr_tensor(
            crow_indices=sparse_mx.indptr,
            col_indices=sparse_mx.indices,
            values=sparse_mx.data,
            size=sparse_mx_shape,
            dtype=dtype,
            device=device,
        )
