
from typing import Optional, List

import pandas as pd
import pytorch_lightning as pl
import pl_bolts 
import segmentation_models_pytorch as smp
import torch
import torchmetrics

# from pytorch_lightning.utilities import rank_zero_only
# from pytorch_lightning.loggers.base import rank_zero_experiment

from .cloud_dataset import CloudDataset
from .losses import intersection_and_union
from .losses import dice_loss, power_jaccard
from .plotting_tools import plot_prediction_grid


class CloudModel(pl.LightningModule):
    def __init__(
        self,
        bands: List[str],
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        cloudbank: Optional[pd.DataFrame] = None,
        train_transforms = None,
        val_transforms = None,
        cloud_transforms = None,
        hparams: dict = {},
    ):
        """
        Instantiate the CloudModel class based on the pl.LightningModule
        (https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).

        Args:
            bands (list[str]): Names of the bands provided for each chip
            x_train (pd.DataFrame, optional): a dataframe of the training features with a row for each chip.
                There must be a column for chip_id, and a column with the path to the TIF for each of bands.
                Required for model training
            y_train (pd.DataFrame, optional): a dataframe of the training labels with a for each chip
                and columns for chip_id and the path to the label TIF with ground truth cloud cover.
                Required for model training
            x_val (pd.DataFrame, optional): a dataframe of the validation features with a row for each chip.
                There must be a column for chip_id, and a column with the path to the TIF for each of bands.
                Required for model training
            y_val (pd.DataFrame, optional): a dataframe of the validation labels with a for each chip
                and columns for chip_id and the path to the label TIF with ground truth cloud cover.
                Required for model training
            cloudbank (pd.DataFrame, optional): a dataframe of paths to additional clouds to sample from. 
                Optional for model training, but required if using chips where label_path=='None'
            hparams (dict, optional): Dictionary of additional modeling parameters.
        """
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # required
        self.bands = bands
        self.num_channels = len(bands)
        
        # optional modeling params
        self.segmentation_model = self.hparams.get("segmentation_model", "unet")
        self.encoder_name = self.hparams.get("encoder_name", "efficientnet-b0")
        self.weights = self.hparams.get("weights", None)
        self.decoder_attention_type = self.hparams.get("decoder_attention_type", None)
        self.custom_feature_channels = self.hparams.get("custom_feature_channels", None)
                                                        
        self.loss_function = self.hparams.get("loss_function", "BCE")        
        self.optimizer = self.hparams.get("optimizer", "ADAM")
        self.scheduler = self.hparams.get("scheduler", "PLATEAU")
        
        self.learning_rate = self.hparams.get("learning_rate", 1e-3)
        self.momentum = self.hparams.get("momentum", 0.9)
        self.T_0 = self.hparams.get("T_0", 10)
        self.eta_min = self.hparams.get("eta_min", 1e-5)
        
        self.warmup_epochs = self.hparams.get("max_epochs", 10)
        self.max_epochs = self.hparams.get("max_epochs", 40)

        self.reduce_learning_rate_factor = self.hparams.get("reduce_learning_rate_factor", 0.1)

        self.patience = self.hparams.get("patience", 10)
        self.learning_rate_patience = self.hparams.get("learning_rate_patience", 5)
        self.batch_size = self.hparams.get("batch_size", 8)

        self.num_workers = self.hparams.get("num_workers", 2)
        self.pin_memory = self.hparams.get("pin_memory", True)
        self.persistent_workers = self.hparams.get("persistent_workers", False)
        
        self.gpu = self.hparams.get("gpu", False)
        
        self.log_on_step = self.hparams.get("log_on_step", False)
        self.progress_bar = self.hparams.get("progress_bar", False)
        
        self.plot_validation_images = self.hparams.get("plot_validation_images", True)
        self.num_images_plot = self.hparams.get("num_images_plot", self.batch_size)

        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.cloud_transform = cloud_transforms
        
        # Instantiate datasets, model, and trainer params if provided
        self.train_dataset = CloudDataset(
            x_paths=x_train,
            bands=self.bands,
            y_paths=y_train,
            transforms=self.train_transform,
            cloudbank=cloudbank,
            cloud_transforms=self.cloud_transform,
            custom_feature_channels=self.custom_feature_channels,
        )
        
        self.val_dataset = CloudDataset(
            x_paths=x_val,
            bands=self.bands,
            y_paths=y_val,
            transforms=self.val_transform,
            custom_feature_channels=self.custom_feature_channels,
        )
        
        # define some performance metrics using torchmetrics
        # self.train_accuracy = torchmetrics.Accuracy()
        # self.val_intersection = mymetrics.Intersection()
        self.val_IoU = torchmetrics.IoU(num_classes=2)
        self.train_IoU = torchmetrics.IoU(num_classes=2)

        self.model = self._prepare_model()

    ## Required LightningModule methods ##
    def forward(self, image: torch.Tensor):
        """
        Forward pass
        output of model is (B, 1, H, W), so remove axis=1
        return raw logits in order to use BCEWithLogitsLoss which is more stable than BCE:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        """
        return self.model(image).view(-1, 512, 512)

    def calculate_loss(self, chip, label, preds):
        if self.loss_function.upper()=="BCE":
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(preds, label.float()).mean()
            
        if self.loss_function.upper()=="DICE":
            loss = dice_loss(preds, label)
            
        if self.loss_function.upper()=="JACCARD":
            loss = power_jaccard(preds, label, power_val=1.)

        return loss

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step.

        Args:
            batch (dict): dictionary of items from CloudDataset of the form
                {'chip_id': list[str], 'chip': list[torch.Tensor], 'label': list[torch.Tensor]}
            batch_idx (int): batch number
        """
        if self.train_dataset.data is None:
            raise ValueError(
                "x_train and y_train must be specified when CloudModel is instantiated to run training"
            )

        # Switch on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)

        if self.loss_function == 'BCE':
            loss = self.calculate_loss(x, y, preds)

        preds = torch.sigmoid(preds)
        
        if self.loss_function != 'BCE':
            loss = self.calculate_loss(x, y, preds)

        preds = (preds > 0.5) * 1  # convert to int

        # batch_intersection, batch_union = intersection_and_union(preds, y)
    
        self.train_IoU(preds, y)

        self.log(
            "train_performance", 
            {"iou": self.train_IoU},
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=self.progress_bar,
        )
        self.log(
            "train_loss",
            loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=self.progress_bar,
        )
        
        # keep seperate to use for early stopping
        self.log("train_iou", self.train_IoU, on_step=True, on_epoch=True, prog_bar=self.progress_bar)

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch (dict): dictionary of items from CloudDataset of the form
                {'chip_id': list[str], 'chip': list[torch.Tensor], 'label': list[torch.Tensor]}
            batch_idx (int): batch number
        """
        if self.val_dataset.data is None:
            raise ValueError(
                "x_val and y_val must be specified when CloudModel is instantiated to run validation"
            )

        # Switch on validation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        chip_id = batch["chip_id"]
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        preds = self.forward(x)

        loss = self.calculate_loss(x, y, preds)

        preds = torch.sigmoid(preds)
        preds = (preds > 0.5) * 1  # convert to int

        if self.plot_validation_images:
            # keep to pass to validation_epoch_end and plot
            self.last_x = x
            self.last_y = y
            self.last_pred = preds
            self.last_chip_id = chip_id

        # Log batch IOU
        batch_intersection, batch_union = intersection_and_union(preds, y)
        self.val_IoU(preds, y)

        self.log("val_performance", 
                 {"iou": self.val_IoU},
                 on_step=self.log_on_step, on_epoch=True, prog_bar=self.progress_bar)
                 
        self.log("val_loss", loss, on_step=self.log_on_step, on_epoch=True, prog_bar=self.progress_bar)
        
        # keep seperate to use for early stopping
        self.log("val_iou", self.val_IoU, on_step=True, on_epoch=True, prog_bar=self.progress_bar)

        return {"loss": loss}#, "x": x, "y": y, "pred": preds}

#     def validation_step_end(self, batch_parts):
#         gpu_use = 0
#         # print(batch_parts['x'][gpu_use].size())
#         return {"x": batch_parts["x"][gpu_use], "y": batch_parts["y"][gpu_use], "pred": batch_parts["pred"][gpu_use]}

    # @rank_zero_only
    # @rank_zero_experiment
    
    def validation_epoch_end(self, outputs):
        # idevice = self.last_x.get_device()
        # if idevice == 0:
        # if self.global_rank==0:
        if self.plot_validation_images:
            # self.logger[0].experiment.add_figure("chip_label_prediction", 
            self.logger.experiment.add_figure("chip_label_prediction", 
                                                 plot_prediction_grid(
                                                     self.last_x,
                                                     self.last_y,
                                                     self.last_pred,
                                                     self.last_chip_id,
                                                     custom_feature_channels=self.custom_feature_channels,
                                                     num_images_plot=self.num_images_plot,
                                                 ),
                                              self.current_epoch,
                                             )

        # if batch_idx == 0:
            # print(out)
            # for out in validation_step_outputs[:1]:
            #     # output from each gpu
            #     print(out)
            
    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size|self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.plot_validation_images, # if plotting last batch images ensure full last batch
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def configure_optimizers(self):
        
        if self.optimizer.upper()=="ADAM":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )
            
        if self.optimizer.upper()=="ADAMW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
            )
            # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        if self.optimizer.upper()=="SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
            )
        
        if self.scheduler.upper()=="EXPONENTIAL":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,
            )
            
        if self.scheduler.upper()=="COSINE":
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=self.T_0,
            #     eta_min=self.eta_min,
            # ) 

            scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
            ) 
  

        if self.scheduler.upper()=="PLATEAU":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'max',
                factor=self.reduce_learning_rate_factor,
                patience=self.learning_rate_patience,
            )
            
            return {"optimizer": optimizer, 
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_iou",
                    },
            }
                                       
        return [optimizer], [scheduler]
                
    ## Convenience Methods ##
    def _prepare_model(self):
        
        if self.segmentation_model.upper()=="UNET":
            # Instantiate U-Net model
            unet_model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                in_channels=self.num_channels,
                classes=1,
                decoder_attention_type=self.decoder_attention_type,
            )
            if self.gpu:
                unet_model.cuda()
                
        if self.segmentation_model.upper()=="DEEPLABV3PLUS":
            # Instantiate DeepLabV3Plus model (https://arxiv.org/abs/1802.02611v3)
            unet_model = smp.DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                in_channels=self.num_channels,
                classes=1,
            )
            if self.gpu:
                unet_model.cuda()

        return unet_model
