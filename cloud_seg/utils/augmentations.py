import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

import numpy as np
import sys

class CloudAugmentations:
    def __init__(self, params):

        self.params = params
        self.min_max_crop = params.get('min_max_crop', [int(512//3*2), 512])
        self.aug_prob_soft = params.get('aug_prob_soft', 0.5) # soft augmentations. e.g. rotate, flip, ...
        self.aug_prob_medium = params.get('aug_prob_medium', 0.8) # random rescale, etc
        self.aug_prob_hard = params.get('aug_prob_hard', 0.5) # grid distortion, etc

        self.grid_distort_limit = params.get('grid_distort_limit', 0.3)

        self.sigma_brightness = params.get('sigma_brightness', 0.1)
        self.mean_brightness = params.get('mean_brightness', 1.0)
        self.per_channel_brightness = params.get('per_channel_brightness', False)
        self.uniform_brightness = params.get('uniform_brightness', False)

        # Dictionary to convert between abbreviation and full augmentation string
        self.aug_to_name = {
            'vf': 'VerticalFlip',
            'hf': 'HorizontalFlip',
            'rr': 'RandomRotate90',            
            'tr': 'Transpose',
            'rc': 'RandomSizedCrop',
            'gd': 'GridDistortion',
            'gb': 'GaussianBlur',
            'br': 'Brightness',
            'nr': 'Normalize',
            # 'ss': 'SizeScale',
            # 'gb': 'GaussianBlur',
            # 'gn': 'GaussianNoise',
        }

        self.aug_to_func = {
            'vf': self.add_VerticalFlip,
            'hf': self.add_HorizontalFlip,
            'rr': self.add_RandomRotate90,
            'tr': self.add_Transpose,
            'rc': self.add_RandomSizedCrop,
            'gd': self.add_GridDistortion,
            'gb': self.add_GaussianBlur,
            'br': self.add_Brightness,
            'nr': self.add_Normalize,
            # 'ss': self.add_SizeScale,
            # 'gb': self.add_GaussianBlur,
            # 'gn': self.add_GaussianNoise,
        }
        
        if params['verbose']:
            print("Available augmentations are: ")
            [print(f"{k}: {v}") for k, v in self.aug_to_name.items()]
        
    # Augmentation functions are listed in the order that they (mostly) should be called   
    def add_VerticalFlip(self):
        self.augmentations_names.append(self.aug_to_name['vf'])
        self.augmentations.append(A.VerticalFlip(p=self.aug_prob_soft))

    def add_HorizontalFlip(self):
        self.augmentations_names.append(self.aug_to_name['hf'])
        self.augmentations.append(A.HorizontalFlip(p=self.aug_prob_soft))

    def add_RandomRotate90(self):
        self.augmentations_names.append(self.aug_to_name['rr'])
        self.augmentations.append(A.RandomRotate90(p=self.aug_prob_soft))
        
    def add_Transpose(self):
        self.augmentations_names.append(self.aug_to_name['tr'])
        self.augmentations.append(A.Transpose(p=self.aug_prob_soft))

    def add_RandomSizedCrop(self):
        self.augmentations_names.append(self.aug_to_name['rc'])
        self.augmentations.append(
            A.RandomSizedCrop(
                height=512,
                width=512,
                min_max_height=self.min_max_crop, 
                p=self.aug_prob_medium,
            )
        )
    def add_GridDistortion(self):
        self.augmentations_names.append(self.aug_to_name['gd'])
        self.augmentations.append(
            A.GridDistortion(
                p=self.aug_prob_hard,
                distort_limit=self.grid_distort_limit,
            )
        )

    def add_GaussianBlur(self):
        self.augmentations_names.append(self.aug_to_name['gb'])
        self.augmentations.append(A.GaussianBlur(p=self.aug_prob_hard))

    def add_Brightness(self):
        self.augmentations_names.append(self.aug_to_name['br'])
        self.augmentations.append(
            ModifyBrightness(
                mean_brightness=self.mean_brightness,
                sigma_brightness=self.sigma_brightness,
                uniform=self.uniform_brightness,
                per_channel=self.per_channel_brightness,
                p=self.aug_prob_medium,
            )
        )

    def add_Normalize(self):
        self.augmentations_names.append(self.aug_to_name['nr'])
        self.augmentations.append(
            A.Normalize(
                mean=self.params['band_means'],
                std=self.params['band_stds'],
                max_pixel_value=self.params['max_pixel_value'],
                p=1.0,
            )
        )    
                                  
    def add_augmentations(self, augs_manual: str=None):
        self.augmentations = []
        self.augmentations_names = []

        # split every two characters     
        if augs_manual is None:
            augmentations_use = self.params['augmentations']
        if augs_manual is not None:
            augmentations_use = augs_manual

        augs = [augmentations_use[i:i+2] for i in range(0, len(augmentations_use), 2)]

        for aug in augs:
            if aug not in self.aug_to_func.keys():
                sys.exit(f"Augmentation abbreviation {aug} is not an available key. Choose from", self.aug_to_name.keys())

            self.aug_to_func[aug]() # () required to actually call function                                    

        if self.params['verbose']:
            print(f"\nUsing augmentations \n{self.augmentations_names}\n {self.augmentations}")

        return self.augmentations, self.augmentations_names



class ModifyBrightness(ImageOnlyTransform):
    """
    Scales brightness by multiplying image by a random draw with sigma=sigma_brightness, mean=1
    """
    def __init__(self, mean_brightness=1.0, sigma_brightness=0.1, uniform=False, per_channel=False, 
                 always_apply=False, p=0.5):
        super(ModifyBrightness, self).__init__(always_apply, p)

        self.mean_brightness = mean_brightness
        self.sigma_brightness = sigma_brightness

        self.uniform = uniform
        self.p = p
        self.per_channel = per_channel

    def get_transform_init_args_names(self):
        return ("sigma_brightness",)        
 
    def apply(self, img, **params):
        if np.random.uniform() > self.p:
            # Don't apply
            return img
        
        nsamples = 1
        if self.per_channel:
            nsamples = img.shape[-1]
            
        # draw 'true' noise level of each channel from lognormal fits
        if not self.uniform:
            amplitude_i = np.random.normal(self.mean_brightness, self.sigma_brightness, size=nsamples)
        if self.uniform:
            amplitude_i = np.random.uniform(
                self.mean_brightness-self.sigma_brightness,
                self.mean_brightness+self.sigma_brightness,
                size=nsamples
            )
        
        return img * np.float32(amplitude_i)

