import numpy as np
import torch
import timeit

from data import transforms as T
from data.transforms import DataTransform_N1, DataTransform_Ni
from model.model import Model
from utils import operator, create_net

class R2D2_model(Model):
    def __init__(self, hparams):
        super().__init__(hparams)
        unet = create_net(hparams, self.device)
        if self.hparams.layers == 1:
            self.compute_forward = self.R2D2_forward
            self.unet = unet
        elif self.hparams.layers > 1:
            self.compute_forward = self.R2D2Net_forward
            self.operator = operator(im_size=(self.hparams.im_dim_x, self.hparams.im_dim_y), op_acc='approx')
            self.unet = []
            for i in range(self.hparams.layers):
                setattr(self, f'unet{i+1}', unet[i])
                self.unet.append(getattr(self, f'unet{i+1}'))
        self.DataTransform = DataTransform_N1 if self.hparams.num_iter == 1 else DataTransform_Ni
                
        
    def forward(self, batch, stage):
        if self.hparams.num_iter == 1:
            _, _, target, _, _, fname, slice, _, a_expo = batch
        else:
            _, _, target, _, _, _, _, fname, slice, _, a_expo = batch
        output, time, loss = self.compute_forward(batch)
        if self.hparams.mode == 'test':
            print(f'{fname[0]}: Time: {time:.4f}')
        self.compute_metrics(output, target, a_expo, stage)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if stage == 'train':
           return loss
        elif stage == 'val':
            return output, target
        elif stage == 'test':
            return output, fname, slice
    
    def R2D2_forward(self, batch):
        if self.hparams.num_iter == 1:
            _, dirty_n, _, target_n, _, _, _, mean, _ = batch
            net_input = torch.cat([dirty_n, torch.zeros_like(dirty_n)], dim=1)
        else:
            _, res_n, _, target_n, _, rec_n, _, _, _, mean, _ = batch
            net_input = torch.cat([rec_n, res_n], dim=1)
        start = timeit.default_timer()
        output = self.unet(net_input)
        stop = timeit.default_timer()
        if self.hparams.positivity:
            output = torch.clip(output, min=0, max=None)
        loss = self.loss_fn(output, target_n)
        output *= (mean + 1e-110)
        return output, stop - start, loss
        
    def R2D2Net_forward(self, batch):
        if self.hparams.num_iter == 1:
            dirty, dirty_n, target, _, PSF, _, _, mean, _ = batch
            net_input = torch.cat([dirty_n, torch.zeros_like(dirty_n)], dim=1)
            prev_est_n = torch.zeros_like(dirty_n)
        else:
            res_prev, res_prev_n, target, _, rec, rec_n, PSF, _, _, mean, _ = batch
            net_input = torch.cat([res_prev_n, rec_n], dim=1)
            prev_est_n = rec_n
        start = timeit.default_timer()
        for i in range(self.hparams.layers):
            output = self.unet[i](net_input) + prev_est_n
            output *= (mean + 1e-110)
            if i < (self.hparams.layers - 1):
                if self.hparams.num_iter == 1:
                    res = self.operator.gen_res(dirty, output, PSF=PSF)
                else:
                    res = self.operator.gen_res(res_prev, output-rec, PSF=PSF)
                prev_est_n, mean = T.normalize_instance(output, eps=1e-110)
                res = T.normalize(res, mean, eps=1e-110)
                net_input = torch.cat([res, prev_est_n], dim=1)
        stop = timeit.default_timer()
        if self.hparams.positivity:
            output = torch.clip(output, min=0, max=None)
        loss = self.loss_fn(output, target)
        return output, stop - start, loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, 'train')
        logs = {'loss': loss.item()}
        self.train_logs.append(dict(loss=loss, log=logs))
        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        output, target = self.forward(batch, 'val')
        self.val_logs.append({
                            'output': output,
                            'target': target,
                            })

    def test_step(self, batch, batch_idx):
        output, fname, slice = self.forward(batch, 'test')
        self.test_logs.append({'output': output,
                               'fname': fname,
                               'slice': slice})
        
    def predict_step(self, batch, batch_idx):
        predictions = self.forward(batch, 'test_single')
        return predictions

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        return self.DataTransform()

    def val_data_transform(self):
        return self.DataTransform()

    def test_data_transform(self):
        return self.DataTransform()