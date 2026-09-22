import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader

from .augment import build_eval_transform, build_train_transform, mixup
from .data import TestDataset, TrainDataset
from .edge_transform import EdgeTransform
from .model import ShapeNet


def train_one_model(
    all_paths,
    all_labels,
    device,
    model_type="rgb",
    img_size=128,
    epochs=35,
    lr=1e-3,
    patience_limit=8,
    pos_weight=None,
    use_mixup=False,
    batch_size=96,
    swa_window=None,
):
    is_edge = model_type == "edge"
    edge_tf = EdgeTransform() if is_edge else None

    train_transform = build_train_transform(model_type, img_size)
    eval_transform = build_eval_transform(img_size)

    n = len(all_paths)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size = int(0.15 * n)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = TrainDataset(
        [all_paths[i] for i in train_idx], [all_labels[i] for i in train_idx], train_transform, edge_tf
    )
    val_ds = TrainDataset(
        [all_paths[i] for i in val_idx], [all_labels[i] for i in val_idx], eval_transform, edge_tf
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    model = ShapeNet(in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_acc, best_state, patience = 0.0, None, 0
    swa_model = None
    swa_start = (epochs - swa_window + 1) if swa_window else None

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            if use_mixup:
                imgs, labels = mixup(imgs, labels, alpha=0.2)
            optimizer.zero_grad()
            criterion(model(imgs).squeeze(1), labels).backward()
            optimizer.step()
        scheduler.step()

        if swa_start is not None and epoch >= swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model)
            else:
                swa_model.update_parameters(model)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(device)).squeeze(1)
                correct += ((logits > 0).long() == labels.to(device).long()).sum().item()
                total += len(labels)
        val_acc = correct / total

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            tag = " ***"
        else:
            patience += 1

        if epoch % 5 == 0 or tag:
            print(f"  [{model_type}] Epoch {epoch:2d}/{epochs}  val={val_acc:.4f}{tag}")

        if patience >= patience_limit:
            print(f"  [{model_type}] Early stop at epoch {epoch} (best={best_val_acc:.4f})")
            break

    if swa_model is not None:
        # SWA weight averaging doesn't capture BatchNorm running stats -
        # they must be recomputed with a forward pass over training data.
        update_bn(train_loader, swa_model, device=device)
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in swa_model.state_dict().items() if "n_averaged" not in k}
        )
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(device)).squeeze(1)
                correct += ((logits > 0).long() == labels.to(device).long()).sum().item()
                total += len(labels)
        swa_val = correct / total
        print(f"  [{model_type}] SWA val: {swa_val:.4f} (best single epoch: {best_val_acc:.4f})")
    else:
        model.load_state_dict(best_state)
    print(f"  [{model_type}] Best val accuracy: {best_val_acc:.4f}")
    return model, eval_transform, edge_tf


def adapt_bn(model, test_paths, eval_transform, edge_tf, device, batch_size=96):
    """Re-estimates BatchNorm running stats on the (unlabeled) test distribution
    to correct for train/test covariate shift before inference."""
    ds = TestDataset(test_paths, eval_transform, edge_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"))

    bn_mods = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_mods:
        m.train()
        m.reset_running_stats()

    with torch.no_grad():
        for imgs in loader:
            model(imgs.to(device))

    for m in bn_mods:
        m.eval()
