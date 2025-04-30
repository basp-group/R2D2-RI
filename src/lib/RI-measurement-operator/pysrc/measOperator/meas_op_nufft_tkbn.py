"""
Measurement operators built on NUFFT using torchkbnufft library
"""

from typing import Tuple

import torch
import torchkbnufft as tkbn

from .meas_op_nufft import MeasOpNUFFT


class MeasOpTkbNUFFT(MeasOpNUFFT):
    """
    Measurement operators built on NUFFT. Use torchkbnufft as the NUFFT package.
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
        mode: str = "table",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        cal_time: bool = False,
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
        :param mode: The mode of the NUFFT, can be "table" or "matrix", defaults to "table"
        :type mode: str, optional
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
        :param cal_time: Enable to compute each forward and adjoint operator timing, defaults to False
        :type cal_time: bool, optional
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
        
        if grid_size is None:
            grid_size = tuple(np.array(self._img_size) * 2)

        # forward operator
        self._nufft_obj = tkbn.KbNufft(
            im_size=self._img_size,
            grid_size=grid_size,
            numpoints=kernel_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # adjoint operator
        self._adj_nufft_obj = tkbn.KbNufftAdjoint(
            im_size=self._img_size,
            grid_size=grid_size,
            numpoints=kernel_dim,
            dtype=self._dtype,
            device=self._device,
        )

        if mode == "matrix":
            self._interp_mats = tkbn.calc_tensor_spmatrix(
                self._traj.squeeze(),
                self._img_size,
                grid_size=grid_size,
                numpoints=kernel_dim,
            )
            self._interp_mats = tuple(
                [t.to(device=self._device, dtype=self._dtype) for t in self._interp_mats]
            )
        elif mode == "table":
            self._interp_mats = None
        else:
            raise NotImplementedError("Unrecognised mode: " + mode)

    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight

        :param x: The input tensor in the image domain.
        :type x: torch.Tensor
        :return: The result of the forward NUFFT operation.
        :rtype: torch.Tensor
        """
        return (
            self._nufft_obj(x.to(self._dtype_meas), self._traj, interp_mats=self._interp_mats)
            * self._data_weight
        )

    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint NUFFT with data weight.

        :param y: The input tensor in the Fourier domain.
        :type y: torch.Tensor
        :return: The result of the adjoint NUFFT operation.
        :rtype: torch.Tensor
        """
        return self._adj_nufft_obj(y * self._data_weight, self._traj, interp_mats=self._interp_mats)
