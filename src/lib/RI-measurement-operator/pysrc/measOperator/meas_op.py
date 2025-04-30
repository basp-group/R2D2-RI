"""
Base class for measurement operator
"""

from abc import ABC, abstractmethod
from functools import wraps

import torch


class MeasOp(ABC):
    """
    Base class for measurement operator
    """

    def __init__(self, img_size, real_flag=True, device=None, dtype=torch.float, cal_time=False):
        self._img_size = img_size
        self._real_flag = real_flag
        self._dtype = dtype
        # Set the data type of the complex measurements
        if self._dtype in [torch.float, torch.float32]:
            self._dtype_meas = torch.complex64
        else:
            self._dtype_meas = torch.complex128

        self._device = device
        self._op_norm = None
        self._cal_time = cal_time

    @torch.no_grad()
    def time_decorator(self, func):
        import time

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__.replace('_inner', '')} took {end - start} seconds")
            return result

        return wrapper

    @torch.no_grad()
    def conditional_decorator(self, decorator, condition):
        def wrapper(func):
            if condition:
                return decorator(func)
            return func

        return wrapper

    @abstractmethod
    def forward_op(self, x):
        """
        Abstract method for forward measurement oprator
        """
        return NotImplemented

    @abstractmethod
    def adjoint_op(self, y):
        """
        Abstract method for adjoint measurement oprator
        """
        return NotImplemented

    def set_real_flag(self, real_flag):
        """
        Set real flag for adjoint measurement oprator
        """
        self._real_flag = real_flag
        self._op_norm = None

    def get_device(self):
        """
        Return the device that the measurement oprator is running on
        """
        return self._device

    def get_data_type(self):
        """
        Return the data type of the target image
        """
        return self._dtype

    def get_data_type_meas(self):
        """
        Return the data type of the measurements
        """
        return self._dtype_meas

    def get_img_size(self):
        """
        Return the iamge size of the target image
        """
        return self._img_size

    @torch.no_grad()
    def get_op_norm(self, compute_flag=False, rel_tol=1e-5, max_iter=500, verbose=False):
        """
        Get the spectral norm of the measurement operator
        """
        if self._op_norm is None or compute_flag:
            x = torch.randn(1, 1, *self._img_size, device=self._device, dtype=self._dtype)
            x = x / torch.linalg.vector_norm(x).item()
            init_val = 1.0

            for k in range(max_iter):
                x = self.adjoint_op(self.forward_op(x))
                val = torch.linalg.vector_norm(x)
                rel_var = abs(val - init_val) / init_val
                if rel_var < rel_tol:
                    break
                init_val = val
                x = x / val
                if verbose:
                    print(f"Iter = {k}, norm = {val.item()}", flush=True)
            self._op_norm = val.item()

        return self._op_norm

    @torch.no_grad()
    def get_psf(self):
        """
        Get the point spread function of the measurement operator
        """
        dirac = torch.zeros(1, 1, *self._img_size, dtype=self._dtype, device=self._device)
        dirac[0, 0, self._img_size[0] // 2, self._img_size[1] // 2] = 1.0
        psf = self.adjoint_op(self.forward_op(dirac))

        return psf
