
# Adapted from https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/callbacks/vision/confused_logit.py#L20-L167
from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, nn

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# from pytorch_lightning.utilities import rank_zero_only
# @rank_zero_only
class DisplayChipsCallback(Callback):  # pragma: no cover
    """Takes the input chip, true label, and label prediction
        trainer = Trainer(callbacks=[DisplayChips()])
    .. note:: Whenever called, this model will look for ``self.last_batch`` and ``self.last_logits``
              in the LightningModule.
    """

    def __init__(
        self,
        num_images_plot: int=4,
    ):
        """
        Args:
            top_k: How many  images we should plot
   
        """
        super().__init__()
        self.num_images_plot = num_images_plot

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        # outputs: Sequence,
        # batch: Sequence,
        # batch_idx: int,
        # dataloader_idx: int,
    ) -> None:
        # show images only every 20 batches
        # if batch_idx != 0:
        #     return

        # pick the last batch and logits
        # x, y = batch["chip"], batch["label"]
        try:
            x = pl_module.last_x.to("cpu")
            y = pl_module.last_y.to("cpu")
            pred = pl_module.last_pred.to("cpu")
            
        except AttributeError as err:
            m = """please track the last_pred in the validation_step like so:
                def validation_step(...):
                    self.last_pred = your_pred
            """
            raise AttributeError(m) from err

        print(pred)
        self._plot(x, y, pred, trainer, pl_module)

    def _plot(
        self,
        x: Tensor,
        y: Tensor,
        pred: Tensor,
        trainer: Trainer,
        model: LightningModule,
    ) -> None:

        batch_size, c, w, h = x.size()

        # final batch may not be full size
        nimg_plt = self.min(batch_size, self.num_images_plot)
        
        fig, axarr = plt.subplots(nrows=nimg_plt, ncols=3, figsize=(15, 5*))
       
        for img_i in range(nimg_plt):
            xi = x[img_i].to("cpu")
            yi = y[img_i].to("cpu")
            predi = pred[img_i].to("cpu")
            
            self.__draw_data_sample(fig, axarr, img_i, 0, xi[0], "Chip")
            self.__draw_label_sample(fig, axarr, img_i, 1, yi, "True label")
            self.__draw_label_sample(fig, axarr, img_i, 2, predi, "Prediction")
            
        # model.logger.experiment.add_figure("validation_predictions", fig, global_step=trainer.global_step)
        # trainer.logger.experiment[0].add_image("validation_predictions", fig, global_step=trainer.global_step)
        # model.log("validation_predictions", fig, global_step=trainer.global_step)

    @staticmethod
    def __draw_data_sample(fig: Figure, axarr: Axes, row_idx: int, col_idx: int, img: Tensor, title: str) -> None:
        im = axarr[row_idx, col_idx].imshow(img)
        axarr[row_idx, col_idx].set_title(title, fontsize=20)
        
    @staticmethod
    def __draw_label_sample(fig: Figure, axarr: Axes, row_idx: int, col_idx: int, img: Tensor, title: str) -> None:
        im = axarr[row_idx, col_idx].imshow(img, vmin=0., vmax=1.)
        axarr[row_idx, col_idx].set_title(title, fontsize=20)
        
                
