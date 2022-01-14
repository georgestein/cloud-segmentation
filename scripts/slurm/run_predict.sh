#!/bin/bash
srun python predict_unet.py --local_run --model_name  epoch=21-val_iou_epoch=0.84.ckpt --load_checkpoint --batch_size 12 --gpu
#srun python predict_unet.py --local_run --model_name test --load_checkpoint --gpu
