"""
Measurement operators built on NUFFT
"""

import gc
from typing import Tuple

import torch

from .meas_op_nufft import MeasOpNUFFT
from .meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft


class MeasOpPSF(MeasOpNUFFT):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Use PSF with 2x FoV to compute residual dirty image. Such PSF is computed
    using the pytorch_finufft package.
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
        normalise_psf: bool = True,
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
        :param normalise_psf: Enable to normalise the PSF with 2x FoV by its max, defaults to True
        :type normalise_psf: bool, optional
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

        # create the PSF with 2x FoV
        self.meas_op2 = MeasOpPytorchFinufft(
            u=u.to(dtype=self._dtype, device=self._device),
            v=v.to(dtype=self._dtype, device=self._device),
            img_size=tuple(i * 2 for i in img_size),
            natural_weight=natural_weight.to(dtype=self._dtype_meas, device=self._device),
            image_weight=image_weight.to(dtype=self._dtype_meas, device=self._device),
            precond_weight=precond_weight.to(dtype=self._dtype, device=self._device),
            real_flag=real_flag,
            device=device,
            dtype=dtype,
            cal_time=cal_time,
        )

        _psf = self.meas_op2.get_psf().view(1, 1, *tuple(i * 2 for i in img_size))
        if normalise_psf:
            _psf /= _psf.max()
        self._psf_fft = torch.fft.fft2(_psf)

        del self.meas_op2
        gc.collect()

        # use FINUFFT to compute back-projected data (dirty image) and PSF peak
        self.meas_op = MeasOpPytorchFinufft(
            u=u.to(dtype=self._dtype, device=self._device),
            v=v.to(dtype=self._dtype, device=self._device),
            img_size=img_size,
            natural_weight=natural_weight.to(dtype=self._dtype_meas, device=self._device),
            image_weight=image_weight.to(dtype=self._dtype_meas, device=self._device),
            precond_weight=precond_weight.to(dtype=self._dtype, device=self._device),
            real_flag=real_flag,
            device=device,
            dtype=dtype,
            cal_time=cal_time,
        )
        
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        pass

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward measurement operator.

        :param x: The input tensor in image domain.
        :type x: torch.Tensor

        :return: The result of the forward measurement operator with natural, image and preconditioning weights.
        :rtype: torch.Tensor
        """

        @self.conditional_decorator(self.time_decorator, self._cal_time)
        def forward_op_inner(x):
            return x

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
        def adjoint_op_inner(y):
            y_fft = torch.fft.fft2(y, s=self._psf_fft.shape[-2:], dim=(-2, -1))
            res = torch.fft.ifft2(y_fft * self._psf_fft, dim=(-2, -1))[
                ..., self._img_size[0] : self._img_size[0] * 2, self._img_size[0] : self._img_size[0] * 2
            ]
            if self._real_flag:
                return torch.real(res)

            return res

        return adjoint_op_inner(y)
