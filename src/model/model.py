from collections import defaultdict
import lightning as L
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from utils.evaluate import SNR_Metric
from utils.io import save_reconstructions, Data_N1, Data_Ni
from data.transforms import to_log

class Model(L.LightningModule):
    """
    Abstract super class for Deep Learning based reconstruction models.
    This is a subclass of the LightningModule class from pytorch_lightning, with
    some additional functionality:
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and implement the
    following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation and testing respectively
        - configure_optimizers:
            Create and return the optimizers
    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_fn = F.l1_loss
        if self.hparams.mode == 'train':
            self.train_logs = []
            self.val_logs = []
        elif 'test' in self.hparams.mode:
            self.test_logs = []
            
        if self.hparams.mode == 'train':
            self.stages = ['train', 'val']
        else:
            self.stages = [self.hparams.mode]
        self.metrics = []
        for stage in self.stages:
            setattr(self, f'{stage}_SNR', SNR_Metric())
            setattr(self, f'{stage}_logSNR', SNR_Metric(log=True))
            setattr(self, f'{stage}_ssim', SSIM())
            setattr(self, f'{stage}_PSNR', PSNR())
        self.metrics = ['SNR', 'logSNR', 'ssim', 'PSNR']

    def _create_data_loader(self, data_transform, data_partition):
        Data = Data_N1 if self.hparams.num_iter == 1 else Data_Ni
        dataset = Data(
            hparams=self.hparams,
            data_partition=data_partition,
            transform=data_transform
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=None,
        )

    def train_data_transform(self):
        raise NotImplementedError

    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition=self.hparams.scname_train)

    def val_data_transform(self):
        raise NotImplementedError

    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition=self.hparams.scname_val)

    def test_data_transform(self):
        raise NotImplementedError

    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(), data_partition=self.hparams.scname_test)
    
    def compute_metrics(self, output, target, a_expo, stage):
        if a_expo > 0:
            metric_list = self.metrics
        else:
            metric_list = [metric for metric in self.metrics if metric != 'logSNR']
        for metric in metric_list:
            if metric == 'logSNR':
                getattr(self, f'{stage}_{metric}')(output, target, a_expo)
            else:
                getattr(self, f'{stage}_{metric}')(output, target)
            self.log(f'{stage}_{metric}', getattr(self, f'{stage}_{metric}'), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    def _evaluate(self, logs, stage):
        outputs = defaultdict(list)
        for log in logs:
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
            
        if self.hparams.save_output and 'test' in stage:
            outpth = str(self.hparams.data_path) +'/'+ str(self.hparams.scname_test) + str(self.hparams.rec2_ext)
            print(f'Saving outputs in {outpth}')
            save_reconstructions(outputs, outpth,
                                 self.hparams.rec_file_ext, self.hparams.res_file_ext)
        metrics = {metric: getattr(self, f'{stage}_{metric}') for metric in self.metrics}
        return dict(log=metrics, **metrics)

    def _visualize(self, val_logs):
        def _normalize(image):
            return (image - image.min()) / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid)

        # Only process first size to simplify visualization.
        visualize_size = val_logs[0]['output'].shape
        val_logs = [x for x in val_logs if x['output'].shape == visualize_size]
        num_logs = len(val_logs)
        num_viz_images = 6
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []
        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_logs[i]['output'][0]))
            targets.append(_normalize(val_logs[i]['target'][0]))
        outputs = torch.stack(outputs)
        targets = torch.stack(targets)
        _save_image(to_log(targets), 'Target (log)')
        _save_image(to_log(outputs), 'Reconstruction (log)')
        _save_image(torch.abs(targets - outputs), 'Error (abs)')

    def on_training_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_logs]).mean()
        self.logger.experiment.add_scalar("Training_loss", avg_loss, self.current_epoch)
        self.train_logs.clear()

    def on_validation_epoch_end(self):
        self._visualize(self.val_logs)
        self.val_logs.clear()

    def on_test_epoch_end(self):
        self._evaluate(self.test_logs, 'test')
