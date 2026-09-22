import glob
import os

from PIL import Image
from torch.utils.data import Dataset


def resolve_dir(data_dir: str, subdir: str) -> str:
    """Handles Kaggle's occasional doubly-nested extraction layout."""
    nested = os.path.join(data_dir, subdir, subdir)
    if os.path.isdir(nested):
        return nested
    return os.path.join(data_dir, subdir)


def load_train_paths(train_dir: str):
    paths, labels = [], []
    for label in (0, 1):
        class_dir = os.path.join(train_dir, str(label))
        for p in sorted(glob.glob(os.path.join(class_dir, "*.png"))):
            paths.append(p)
            labels.append(label)
    return paths, labels


class TrainDataset(Dataset):
    def __init__(self, paths, labels, transform, edge_transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.edge_transform = edge_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.edge_transform:
            img = self.edge_transform(img)
        return self.transform(img), self.labels[idx]


class TestDataset(Dataset):
    def __init__(self, paths, transform, edge_transform=None):
        self.paths = paths
        self.transform = transform
        self.edge_transform = edge_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.edge_transform:
            img = self.edge_transform(img)
        return self.transform(img)
