"""
Measurement operators built on NUFFT
"""

from abc import abstractmethod
from typing import Tuple

import torch

from .meas_op import MeasOp


class MeasOpNUFFT(MeasOp):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Parent class for NUFFT based measurement operators.
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
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
        :param cal_time: Enable to compute each forward and adjoint operator timing, defaults to False
        :type cal_time: bool, optional
        """
        super().__init__(
            img_size=img_size,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
            cal_time=cal_time,
        )
        self._traj = torch.cat((v, u), dim=1).to(dtype=self._dtype, device=self._device)

        self._natural_weight = natural_weight.to(dtype=self._dtype_meas, device=self._device)
        self._image_weight = image_weight.to(dtype=self._dtype_meas, device=self._device)

        self._data_weight = (self._natural_weight * self._image_weight).to(
            dtype=self._dtype_meas, device=self._device
        )
        self._precond_weight = precond_weight.to(dtype=self._dtype_meas, device=self._device)
        self._grid_size = None
        self._kernel_dim = None
        self._mode = None

    @abstractmethod
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight
        Different NUFFT packages have different implementations
        """
        return NotImplemented

    @abstractmethod
    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight
        Different NUFFT packages have different implementations
        """
        return NotImplemented

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward measurement operator.

        :param x: The input tensor in image domain.
        :type x: torch.Tensor

        :return: The result of the forward measurement operator with natural, image and preconditioning weights.
        :rtype: torch.Tensor
        """

        @self.conditional_decorator(self.time_decorator, self._cal_time)
        def forward_op_inner(x: torch.Tensor) -> torch.Tensor:
            return self._GA(x) * self._precond_weight

        return forward_op_inner(x)

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint measurement operator.

        :param y: The input tensor in Fourier domain.
        :type y: torch.Tensor

        :return: The result of the adjoint measurement operator with natural, image and preconditioning weights.
        :rtype: torch.Tensor
        """

        @self.conditional_decorator(self.time_decorator, self._cal_time)
        def adjoint_op_inner(y: torch.Tensor) -> torch.Tensor:
            if self._real_flag:
                return torch.real(self._AtGt(y * self._precond_weight))

            return self._AtGt(y * self._precond_weight)

        return adjoint_op_inner(y)

    def set_precond_weight(self, precond_weight: torch.Tensor) -> None:
        """
        Set preconditioning weight for measurement operator.

        :param precond_weight: The preconditioning weight tensor.
        :type precond_weight: torch
        """
        self._precond_weight = precond_weight.to(dtype=self._dtype_meas, device=self._device)
        self._op_norm = None

    @torch.no_grad()
    def get_op_norm_prime(self, rel_tol: float = 1e-5, max_iter: int = 500, verbose: bool = False) -> float:
        """
        Get the spectral norm of measurement operator with data weight applied twice.
        It will be used to calculated the correction factor of the heuristic noise level.

        :param rel_tol: The relative tolerance. Defaults to 1e-5.
        :type rel_tol: float, optional
        :param max_iter: The maximum number of iterations. Defaults to 500.
        :type max_iter: int, optional
        :param verbose: If True, print progress messages. Defaults to False.
        :type verbose: bool, optional

        :return: The spectral norm of the measurement operator with data weight applied twice.
        :rtype: float
        """
        if not torch.allclose(self._image_weight, torch.ones_like(self._image_weight)):
            precond_weight = self._precond_weight
            op_norm = self._op_norm
            self._op_norm = None

            self._precond_weight = self._image_weight  # apply image weight twice
            op_norm_prime = self.get_op_norm(rel_tol=rel_tol, max_iter=max_iter, verbose=verbose)
            self._precond_weight = precond_weight
            self._op_norm = op_norm
        else:
            op_norm_prime = self.get_op_norm()

        return op_norm_prime
