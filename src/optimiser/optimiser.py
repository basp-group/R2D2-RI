"""
Base class for optimisers
"""

import os
from abc import ABC, abstractmethod
import torch
from astropy.io import fits


class Optimiser(ABC):
    """
    Base class for optimisers
    """

    def __init__(self, 
                 meas, 
                 meas_op, 
                 save_pth="results"):
        self._meas_op = meas_op
        self._meas = meas.to(self._meas_op.get_data_type_meas())
        self._save_pth = save_pth

        # common initialisation
        if "psf" in str(meas_op.__class__.__name__).lower():
            self._psf = self._meas_op.meas_op.get_psf()
            self._psf_peak = torch.amax(self._psf, dim=(-2, -1), keepdim=True).to(self._meas_op._dtype)
            self._meas_bp = self._meas_op.meas_op.adjoint_op(self._meas).to(
                self._meas_op.get_device()
            ) / self._psf_peak
        else:
            self._psf = self._meas_op.get_psf()
            self._psf_peak = torch.amax(self._psf, dim=(-2, -1), keepdim=True).to(self._meas_op._dtype)
            self._meas_bp = self._meas_op.adjoint_op(self._meas).to(
                self._meas_op.get_device()
            ) / self._psf_peak
        self._model = torch.zeros_like(self._meas_bp)
        self._model_prev = torch.zeros_like(self._meas_bp)
        self._sigma_res = torch.tensor([1.0], dtype=self._meas_op._dtype).to(self._meas_op.get_device())
        self._sigma_res_prev = self._sigma_res

        # save dirty image and psf
        fits.writeto(
            os.path.join(self._save_pth, "dirty_normalised.fits"),
            self._meas_bp.squeeze().float().cpu().numpy(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(self._save_pth, "PSF.fits"),
            self._psf.squeeze().float().cpu().numpy(),
            overwrite=True,
        )

    @abstractmethod
    def initialisation(self):
        """
        Optimiser initialisation, should be called before run
        """
        return NotImplemented

    @abstractmethod
    def run(self):
        """
        Run the main loop of the optimiser
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_begin(self):
        """
        Will be called at the beginning of each interation
        """
        return NotImplemented

    @abstractmethod
    def _stop_criteria(self):
        """
        Check the stop criteria
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_end(self):
        """
        Will be called at the beginning of each interation
        """
        return NotImplemented

    @abstractmethod
    def finalisation(self):
        """
        Should be called after the finishing of the optimiser's main loop
        """
        return NotImplemented