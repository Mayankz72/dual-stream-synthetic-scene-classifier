import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import TestDataset


def predict_with_tta(model, test_paths, eval_transform, edge_tf, device, img_size=128, batch_size=96):
    """Averages logits over 4 deterministic views: identity, H-flip, V-flip,
    both flips. Scenes have no canonical orientation, so this reduces variance
    without needing stochastic augmentation at inference time."""
    tta_transforms = [
        eval_transform,
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
    ]

    model.eval()
    all_logits = np.zeros(len(test_paths))

    for tf in tta_transforms:
        ds = TestDataset(test_paths, tf, edge_tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
        idx = 0
        with torch.no_grad():
            for imgs in loader:
                logits = model(imgs.to(device)).squeeze(1).cpu().numpy()
                all_logits[idx : idx + len(logits)] += logits
                idx += len(logits)

    return all_logits / len(tta_transforms)


def select_pseudo_labels(rgb_logits, edge_logits, test_paths, neg_thresh=0.15):
    """Conservative negative-only pseudo-labeling: a test image is pseudo-labeled
    0 only when BOTH streams agree it's confidently negative. A false positive
    pseudo-label would inflate the positive rate in round 2, so we never emit
    positive pseudo-labels."""
    rgb_sig = 1.0 / (1.0 + np.exp(-rgb_logits))
    edge_sig = 1.0 / (1.0 + np.exp(-edge_logits))
    both_neg = (rgb_sig < neg_thresh) & (edge_sig < neg_thresh)
    neg_idx = np.where(both_neg)[0]
    print(f"  Confident negatives (rgb<{neg_thresh} & edge<{neg_thresh}): {len(neg_idx)}")

    paths = [test_paths[i] for i in neg_idx]
    labels = [0] * len(neg_idx)
    return paths, labels
