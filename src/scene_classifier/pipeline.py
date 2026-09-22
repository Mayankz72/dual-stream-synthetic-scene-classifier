import argparse
import glob
import os

import pandas as pd
import torch

from .data import load_train_paths, resolve_dir
from .inference import predict_with_tta, select_pseudo_labels
from .train import adapt_bn, train_one_model


def generate_predictions(data_dir: str, output_path: str = "submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir = resolve_dir(data_dir, "train")
    all_paths, all_labels = load_train_paths(train_dir)
    print(f"Train: {len(all_paths)} images ({all_labels.count(0)} neg, {all_labels.count(1)} pos)")

    test_dir = resolve_dir(data_dir, "test")
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    print(f"Test: {len(test_paths)} images")

    IMG_SIZE = 192
    BATCH_SIZE = 96
    POS_WEIGHT = 0.6
    RGB_EPOCHS = 60
    EDGE_EPOCHS = 40
    SWA_RGB = 15
    SWA_EDGE = 10

    common_kw = dict(img_size=IMG_SIZE, lr=1e-3, batch_size=BATCH_SIZE, pos_weight=POS_WEIGHT)

    # --- Round 1: train on labeled data only ---
    print("\n=== Round 1: RGB model ===")
    rgb_model, rgb_eval_tf, _ = train_one_model(
        all_paths, all_labels, device, model_type="rgb",
        epochs=RGB_EPOCHS, patience_limit=RGB_EPOCHS,
        use_mixup=True, swa_window=SWA_RGB, **common_kw,
    )

    print("\n=== Round 1: Edge model ===")
    edge_model, edge_eval_tf, edge_tf = train_one_model(
        all_paths, all_labels, device, model_type="edge",
        epochs=EDGE_EPOCHS, patience_limit=EDGE_EPOCHS,
        use_mixup=False, swa_window=SWA_EDGE, **common_kw,
    )

    # --- Pseudo-labeling: confident negatives only ---
    print("\n=== Generating pseudo-labels (negatives only) ===")
    adapt_bn(rgb_model, test_paths, rgb_eval_tf, None, device, BATCH_SIZE)
    adapt_bn(edge_model, test_paths, edge_eval_tf, edge_tf, device, BATCH_SIZE)
    rgb_logits_r1 = predict_with_tta(rgb_model, test_paths, rgb_eval_tf, None, device, IMG_SIZE, BATCH_SIZE)
    edge_logits_r1 = predict_with_tta(edge_model, test_paths, edge_eval_tf, edge_tf, device, IMG_SIZE, BATCH_SIZE)
    pseudo_paths, pseudo_labels = select_pseudo_labels(rgb_logits_r1, edge_logits_r1, test_paths, neg_thresh=0.15)

    del rgb_model, edge_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combined_paths = all_paths + pseudo_paths
    combined_labels = all_labels + pseudo_labels
    n_pos = sum(combined_labels)
    print(f"  Combined train set: {len(combined_paths)} images ({n_pos} pos, {len(combined_labels) - n_pos} neg)")

    # --- Round 2: retrain on labeled + pseudo-labeled data ---
    print("\n=== Round 2: RGB model (with neg pseudo-labels) ===")
    rgb_model, rgb_eval_tf, _ = train_one_model(
        combined_paths, combined_labels, device, model_type="rgb",
        epochs=RGB_EPOCHS, patience_limit=RGB_EPOCHS,
        use_mixup=True, swa_window=SWA_RGB, **common_kw,
    )

    print("\n=== Round 2: Edge model (with neg pseudo-labels) ===")
    edge_model, edge_eval_tf, edge_tf = train_one_model(
        combined_paths, combined_labels, device, model_type="edge",
        epochs=EDGE_EPOCHS, patience_limit=EDGE_EPOCHS,
        use_mixup=False, swa_window=SWA_EDGE, **common_kw,
    )

    print("\n=== BN test-time adaptation ===")
    adapt_bn(rgb_model, test_paths, rgb_eval_tf, None, device, BATCH_SIZE)
    adapt_bn(edge_model, test_paths, edge_eval_tf, edge_tf, device, BATCH_SIZE)

    print("\n=== Final inference ===")
    rgb_logits = predict_with_tta(rgb_model, test_paths, rgb_eval_tf, None, device, IMG_SIZE, BATCH_SIZE)
    edge_logits = predict_with_tta(edge_model, test_paths, edge_eval_tf, edge_tf, device, IMG_SIZE, BATCH_SIZE)

    ensemble_logits = 0.5 * rgb_logits + 0.5 * edge_logits
    preds = (ensemble_logits > 0).astype(int)

    ids = [os.path.basename(p) for p in test_paths]
    df = pd.DataFrame({"ID": ids, "Label": preds})
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}  ({preds.sum()} pos / {len(preds)} total)")
    print(f"RGB predicts:  {(rgb_logits > 0).sum()} pos")
    print(f"Edge predicts: {(edge_logits > 0).sum()} pos")


def main():
    parser = argparse.ArgumentParser(description="Dual-stream RGB+Edge ShapeNet training/inference pipeline.")
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/competitions/iith-deep-learning-2026-hackathon",
        help="Root dir containing train/{0,1}/*.png and test/*.png",
    )
    parser.add_argument("--output", default="submission.csv", help="Path to write predictions CSV")
    args = parser.parse_args()
    generate_predictions(args.data_dir, args.output)


if __name__ == "__main__":
    main()
