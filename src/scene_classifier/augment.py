import numpy as np
import torch
from torchvision import transforms


def build_train_transform(model_type: str, img_size: int):
    if model_type == "edge":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.5, 3.0))], p=0.3),
            transforms.RandomApply([transforms.RandomAutocontrast()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ]
    )


def build_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def mixup(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_imgs, mixed_labels
