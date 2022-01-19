import albumentations as A
import numpy as np
import sys

class CloudAugmentations:
    def __init__(self, params):

        self.params = params
        self.min_max_crop = params.get('min_max_crop', [512//2, 512])
        self.aug_prob_soft = params.get('aug_prob_soft', 0.5) # soft augmentations. e.g. rotate, flip, ...
        self.aug_prob_medium = params.get('aug_prob_medium', 0.8) # random rescale, etc
        self.aug_prob_hard = params.get('aug_prob_hard', 0.2) # grid distortion, etc
        
        # Dictionary to convert between abbreviation and full augmentation string                              
        self.aug_to_name = {
            'vf': 'VerticalFlip',
            'hf': 'HorizontalFlip',
            'rr': 'RandomRotate90',            
            'tr': 'Transpose',
            'rc': 'RandomSizedCrop',
            'gd': 'GridDistortion',
            'gb': 'GaussianBlur',
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
        self.augmentations.append(A.GridDistortion(p=self.aug_prob_hard))

    def add_GaussianBlur(self):
        self.augmentations_names.append(self.aug_to_name['gb'])
        self.augmentations.append(A.GaussianBlur(p=self.aug_prob_hard))

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
                sys.exit(f"Augmentation abbreviation {aug} is not an available key. Choose from", self.aug_to_name.key())

            self.aug_to_func[aug]() # () required to actually call function                                    

        if self.params['verbose']:
            print(f"\nUsing augmentations \n{self.augmentations_names}\n {self.augmentations}")

        return self.augmentations, self.augmentations_names
