# import numpy as np
import os
from timeit import default_timer as timer

import torch
from astropy.io import fits

from model import forward
from utils import (
    create_net_imaging,
    get_DNNs,
    load_net,
    normalize,
    normalize_instance,
    read_fits_as_tensor,
    snr_t,
    to_log,
    vprint,
)

from .optimiser import Optimiser


class R2D2(Optimiser):
    def __init__(
        self,
        meas: torch.Tensor,
        meas_op,
        ckpt_path: str,
        layers: int,
        num_iter: int,
        meas_op2=None,
        save_pth: str = "results",
        save_all_outputs: bool = False,
        gdth_file: str = None,
        target_dynamic_range: float = None,
        architecture: str = "unet",
        num_chans: int = 64,
        prune: bool = False,
        sigma_res_tol: float = 1e-4,
        input_order: str = "amir",
        device=torch.device("cpu"),
        verbose=False,
    ):
        """
        R2D2 optimiser

        Parameters
        ----------
        meas_op : class
        """
        super().__init__(meas, meas_op, save_pth)
        self._meas_op2 = meas_op2
        self._ckpt_path = ckpt_path
        self._layers = layers
        self._num_iter = num_iter
        self._device = device
        self._save_all_outputs = save_all_outputs
        self._gdth_file = gdth_file
        self._verbose = verbose
        self._target_dynamic_range = target_dynamic_range
        self._architecture = architecture
        self._num_chan = num_chans
        self._eps = 1e-110
        self._t_iter_model = 0.0
        self._t_iter_residual = 0.0
        self._t_total = 0.0
        self._t_iter = 0.0
        self._sigma_res_tol = sigma_res_tol
        self._prune = prune
        if self._prune:
            self._sigma_res_violation_count = 0
        self._input_order = input_order

        if "psf" in str(meas_op.__class__.__name__).lower():
            self._psf_peak = 1
            
        # cuda event
        self._forward_cuda_timing = False
        self._adjoint_cuda_timing = False
        if self._meas_op.get_device() == torch.device("cuda"):
            self._forward_cuda_timing = True
        if self._meas_op.get_device() == torch.device("cuda"):
            self._adjoint_cuda_timing = True

    @torch.no_grad()
    def initialisation(self):
        self._dnns_dict = get_DNNs(self._num_iter, self._ckpt_path)
        self._dirty_norm = torch.linalg.norm(self._meas_bp, dim=(-2, -1))
        
        if self._layers > 1:
            dirac2 = torch.zeros(1, 1, *self._meas_op2._img_size, dtype=torch.double, device=self._device)
            dirac2[0, 0, self._meas_op2._img_size[0] // 2, self._meas_op2._img_size[1] // 2] = 1.0
            self._psf2 = self._meas_op2.adjoint_op(self._meas_op2.forward_op(dirac2))  # .to(torch.float32)
            fits.writeto(
                os.path.join(self._save_pth, "PSF2.fits"),
                self._psf2.squeeze().float().cpu().numpy(),
                overwrite=True,
            )
            self._psf2 /= torch.amax(self._psf2, dim=(-2, -1), keepdim=True)
        else:
            self._psf2 = None
        
        self._residual, self._mean = normalize_instance(self._meas_bp, eps=self._eps)
        
        self._residual = self._residual.to(torch.float32)
        self._model = self._model.to(torch.float32)
        self._model_prev = self._model_prev.to(torch.float32)
        self._sigma_res = self._sigma_res.to(torch.float32)
        self._sigma_res_prev = self._sigma_res_prev.to(torch.float32)

        self._net = create_net_imaging(
            layers=self._layers,
            num_chans=self._num_chan,
            architecture=self._architecture,
            device=self._device,
        )

        if self._gdth_file is not None:
            self._gdth = read_fits_as_tensor(self._gdth_file).to(self._device)
            if self._target_dynamic_range is not None:
                self._gdth_log = to_log(self._gdth, self._target_dynamic_range)

        vprint("\n*************************************************", self._verbose)
        self._series = "R2D2" if self._layers == 1 else f"R3D3_{self._layers}L"
        vprint(f"********* STARTING ALGORITHM:    {self._series}   *********", self._verbose)
        vprint("*************************************************", self._verbose)

    @torch.no_grad()
    def _each_iter_begin(self):
        self._net = load_net(self._net, self._iter + 1, self._layers, self._dnns_dict)
        if self._iter + 1 > 1:
            # normalize the residual dirty image and the updated image estimate
            self._model, self._mean = normalize_instance(self._model, eps=self._eps)
            self._model = self._model.to(torch.float32)
            self._residual = normalize(self._residual, self._mean, eps=self._eps).to(torch.float32)
        vprint(f"Iteration {self._iter+1}:", self._verbose)

    @torch.no_grad()
    def _each_iter_end(self):
        self._model_prev = self._model.clone()
        vprint(f"Time for model update: {self._t_iter_model:.4f} sec", self._verbose)
        vprint(f"Time for residual computation: {self._t_iter_residual:.4f} sec", self._verbose)
        if self._save_all_outputs and self._iter < self._num_iter - 1:
            fits.writeto(
                os.path.join(self._save_pth, f"tempModel_{str(self._iter + 1)}.fits"),
                self._model.squeeze().float().cpu().numpy(),
                overwrite=True,
            )
            if self._gdth_file is not None:
                snr_tmp = snr_t(self._gdth, self._model, 1e-110)
                vprint(f"SNR: {snr_tmp:.4f} dB", self._verbose)
                if self._target_dynamic_range is not None:
                    log_snr_tmp = snr_t(
                        self._gdth_log, to_log(self._model, self._target_dynamic_range), 1e-110
                    )
                    vprint(f"logSNR: {log_snr_tmp:.4f} dB", self._verbose)
            fits.writeto(
                os.path.join(self._save_pth, f"tempResidual_{str(self._iter + 1)}.fits"),
                self._residual.squeeze().float().cpu().numpy(),
                overwrite=True,
            )

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        self._sigma_res = torch.linalg.norm(self._residual.squeeze(0).squeeze(0)) / self._dirty_norm
        if self._prune:
            print(f"RDR: {self._sigma_res.item():.4e}")
            _sigma_res_rel_diff = abs(self._sigma_res - self._sigma_res_prev) / self._sigma_res_prev
            print(f"RDR rel. diff: {_sigma_res_rel_diff.item():.4e}")

            # Initialize the counter for consecutive sigma_res violations if it doesn't exist
            if not hasattr(self, "_sigma_res_violation_count"):
                self._sigma_res_violation_count = 0

            # Check the stop criteria
            if self._sigma_res > self._sigma_res_prev:
                # Increment the violation count
                self._sigma_res_violation_count += 1
            else:
                # Reset the counter if the condition is not met
                self._sigma_res_violation_count = 0

            # Stop if the condition has been met 2 consecutive times
            stop = (self._sigma_res_violation_count >= 2) or (_sigma_res_rel_diff < self._sigma_res_tol)

            # Update the previous sigma_res for the next iteration
            self._sigma_res_prev = self._sigma_res
            return stop
        else:
            # self._sigma_res = torch.linalg.norm(self._residual.squeeze(0).squeeze(0)) / self._dirty_norm
            print(f"RDR: {self._sigma_res.item():.4e}")
            return False

    @torch.no_grad()
    def run(self):
        # timing with cuda events
        if self._forward_cuda_timing:
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
        if self._adjoint_cuda_timing:
            adjoint_start_event = torch.cuda.Event(enable_timing=True)
            adjoint_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        self._t_total = timer()
        for self._iter in range(self._num_iter):
            self._t_iter = timer()
            self._each_iter_begin()
            if self._forward_cuda_timing:
                forward_start_event.record()
            else:
                self._t_iter_model = timer()
            self._model = forward(
                self._layers,
                self._iter,
                self._net,
                self._residual,
                self._model,
                self._mean,
                self._eps,
                self._meas_bp,
                self._psf2,
                self._input_order,
            )
            if self._forward_cuda_timing:
                forward_end_event.record()
                torch.cuda.synchronize()
                self._t_iter_model = forward_start_event.elapsed_time(forward_end_event) / 1e3
            else:
                self._t_iter_model = timer() - self._t_iter_model
            # compute residual dirty image
            if self._adjoint_cuda_timing:
                adjoint_start_event.record()
            else:
                self._t_iter_residual = timer()
            self._residual = (
                self._meas_bp
                - self._meas_op.adjoint_op(self._meas_op.forward_op(self._model)) / self._psf_peak
            )
            if self._adjoint_cuda_timing:
                adjoint_end_event.record()
                torch.cuda.synchronize()
                self._t_iter_residual = adjoint_start_event.elapsed_time(adjoint_end_event) / 1e3
            else:
                self._t_iter_residual = timer() - self._t_iter_residual
            if self._stop_criteria():
                break
            self._each_iter_end()
            self._t_iter = timer() - self._t_iter
        self._t_total = timer() - self._t_total

    @torch.no_grad()
    def finalisation(self):
        vprint("\n**************************************", self._verbose)
        vprint("********** END OF ALGORITHM **********", self._verbose)
        vprint("**************************************\n", self._verbose)
        vprint(
            f"Imaging finished in {self._iter+1}/{self._num_iter} iterations in {self._t_total} sec, ",
            self._verbose,
        )
        if self._gdth_file is not None:
            snr_tmp = snr_t(self._gdth, self._model, 1e-110)
            vprint(f"SNR: {snr_tmp:.4f} dB", self._verbose)
            if self._target_dynamic_range is not None:
                log_snr_tmp = snr_t(self._gdth_log, to_log(self._model, self._target_dynamic_range), 1e-110)
                vprint(f"logSNR: {log_snr_tmp:.4f} dB", self._verbose)

        residual = (
            self._meas_bp - self._meas_op.adjoint_op(self._meas_op.forward_op(self._model)) / self._psf_peak
        )
        residual = residual.squeeze().double().cpu().numpy()
        fits.writeto(
            os.path.join(self._save_pth, f"{self._series}_model_image.fits"),
            self._model.squeeze().double().cpu().numpy(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(self._save_pth, f"{self._series}_residual_dirty_image.fits"),
            self._residual.squeeze().double().cpu().numpy(),
            overwrite=True,
        )
