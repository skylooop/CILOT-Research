import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



class TrainTransform:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    224, interpolation=2 #inter cubic
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brigtness=0.4, contrast=0.4, hue=0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(p=1.0),
                A.Solarize(p=0.0),
                ToTensorV2(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = A.Compose(
            [
                A.RandomResizedCrop(
                    224, interpolation=2
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brigtness=0.4, contrast=0.4, hue=0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(p=1.0),
                A.Solarize(p=0.0),
                ToTensorV2(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2