"""
Measurement operators built on NUFFT
"""

from typing import Tuple
import torch

from .meas_op import MeasOp


class MeasOpDTFT(MeasOp):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Use Direct-time Fourier Transform, very computationally expensive,
    often infeasible for large data.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        img_size: Tuple[int, int],
        pixel_size: float = 2.e-6,
        natural_weight=torch.ones(1, 1, dtype=torch.complex128),
        image_weight=torch.ones(1, 1, dtype=torch.complex128),
        precond_weight: torch.Tensor = torch.ones(1, 1, dtype=torch.complex128),
        real_flag: bool = True,
        grid_centred: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        cal_time: bool = False,
    ) -> None:
        """
        Initialize the measurement operator.

        :param u: The u coordinates of the sampling points in units of wavelength.
        :type u: torch.Tensor
        :param v: The v coordinates of the sampling points in units of wavelength.
        :type v: torch.Tensor
        :param img_size: The target image size.
        :type img_size: Tuple[int, int]
        :param pixel_size: The pixel size of the image in arcsec, defaults to 2.e-6
        :type pixel_size: float, optional
        :param natural_weight: The natural weight for the measurement operator, defaults to torch.ones(1, 1)
        :type natural_weight: torch.Tensor, optional
        :param image_weight: The image weight for the measurement operator, defaults to torch.ones(1, 1)
        :type image_weight: torch.Tensor, optional
        :param precond_weight:  The preconditioning weight for the measurement operator, defaults to torch.ones(1, 1)
        :type precond_weight: torch.Tensor, optional
        :param real_flag: Flag to indicate the image is real of complex, defaults to True
        :type real_flag: bool, optional
        :param grid_centred: If the grid should be centred at 0, defaults to True
        :type grid_centred: bool, optional
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
        :param cal_time: Enable to compute each forward and adjoint operator timing, defaults to False
        :type cal_time: bool, optional
        """
        super().__init__(img_size, real_flag=real_flag, device=device, dtype=dtype, cal_time=cal_time)
        if self._dtype == torch.float:
            self._dtype_meas = torch.complex64
        else:
            self._dtype_meas = torch.complex128
            
        self._traj = torch.cat((v, u), dim=1).to(dtype=self._dtype, device=self._device)
        self._natural_weight = natural_weight.to(dtype=self._dtype_meas, device=self._device)
        self._image_weight = image_weight.to(dtype=self._dtype_meas, device=self._device)

        self._data_weight = (self._natural_weight * self._image_weight).to(
            dtype=self._dtype_meas, device=self._device
        )
        self._precond_weight = precond_weight.to(dtype=self._dtype, device=self._device)
        self._RADPERAS = 1/ (180 / torch.pi * 3600)
        self._pixel_size = pixel_size * self._RADPERAS # convert to radians
        self._fov = self._img_size[0] * self._pixel_size
        
        if grid_centred:
            # self._x_coords = torch.arange(0, -self._img_size[0], -1, dtype=self._dtype) * self._pixel_size \
            #     + (self._fov) / 2.0 - self._pixel_size / 2.0
            self._x_coords = torch.tensor([i * self._pixel_size + self._pixel_size / 2 - self._fov / 2 for i in range(self._img_size[0])], dtype=self._dtype)
        else:
            self._x_coords = torch.arange(0, self._img_size[0], 1, dtype=self._dtype) * self._pixel_size   
        self._x_coords = self._x_coords.to(dtype=self._dtype, device=self._device)
                
        self._batch_size = self._traj.shape[0]
        self.grid_x, self.grid_y = torch.meshgrid(self._x_coords.flatten(), self._x_coords.flatten())
        self.xy = torch.stack((self.grid_x.flatten(), self.grid_y.flatten()), dim=0)
        self.xy = self.xy.view(1, *self.xy.shape).repeat(self._batch_size, 1, 1).to(self._device)
        self.compute_G()
        
    @torch.no_grad()
    def compute_G(self):
        @self.conditional_decorator(self.time_decorator, self._cal_time)
        def compute_G_inner(self):
            self._M = torch.tensor([max(self._traj.shape)])
            N = torch.tensor([self.grid_x.numel()])
            self._G = torch.zeros(self._M, N, dtype=self._dtype_meas, device=self._device)
            self._G = torch.exp(-2. * torch.pi * 1j * torch.matmul(self._traj.permute(0, 2, 1), self.xy))
            self._G = torch.mul(self._G.permute(0, 2, 1), self._data_weight).permute(0, 2, 1)
        compute_G_inner(self)
        
        # TODO: create G mask to reduce the size of G
    
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight

        :param x: The input tensor in the image domain.
        :type x: torch.Tensor
        :return: The result of the forward NUFFT operation.
        :rtype: torch.Tensor
        """
        x = x.view(self._batch_size, -1, 1).to(self._dtype_meas)
        return torch.matmul(self._G, x).view(self._batch_size, 1, self._M) #* self._data_weight

    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint NUFFT with data weight.

        :param y: The input tensor in the Fourier domain.
        :type y: torch.Tensor
        :return: The result of the adjoint NUFFT operation.
        :rtype: torch.Tensor
        """
        return torch.matmul(self._G.conj().permute(0, 2, 1), y.permute(0, 2, 1)).view(self._batch_size, 1, *self._img_size)
    
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
            return self._GA(x) * self._precond_weight# * self._die
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
            if self._real_flag:
                return torch.real(self._AtGt(y * self._precond_weight))

            return self._AtGt(y * self._precond_weight)
        return adjoint_op_inner(y)