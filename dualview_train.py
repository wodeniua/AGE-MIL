# !/usr/bin/env python
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import csv
from pprint import pprint

import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import engine
from architecture.dual_view import DualViewPatientClassifier
from datasets.dual_view_datasets import collate_patient_dual_list, dual_npy_feat_dataset
from engine import dual_view_evaluate, train_one_epoch
from utils.utils import Struct, save_model, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_arguments():
    parser = argparse.ArgumentParser("Dual-view patient classification training", add_help=False)
    parser.add_argument("--config", dest="config", default="config/config.yml", help="dataset settings in yaml format")
    parser.add_argument("--seed", type=int, default=0, help="set the random seed")
    parser.add_argument("--wandb_mode", default="disabled", choices=["offline", "online", "disabled"], help="wandb mode")
    parser.add_argument("--arch", type=str, default="dual_view", choices=["dual_view"], help="architecture type")
    parser.add_argument("--pretrain", default="titan", choices=[
        "medical_ssl", "natural_supervised", "plip", "path-clip-B-AAAI", "path-clip-B",
        "path-clip-L-336", "openai-clip-B", "openai-clip-L-336", "quilt-net", "biomedclip",
        "path-clip-L-768", "UNI", "GigaPath", "conch_v1", "conch_v1_5", "titan",
    ], help="pretrained backbone")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--train_epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--results_path", type=str, default="zresults/res_dual_arsi.csv", help="path to save results")
    parser.add_argument("--notes", type=str, default="", help="results notes")
    return parser.parse_args()


def set_feature_dims(conf):
    if conf.pretrain == "medical_ssl":
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain in {"natural_supervised", "path-clip-B", "openai-clip-B", "plip", "quilt-net", "path-clip-B-AAAI", "biomedclip", "conch_v1"}:
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain in {"path-clip-L-336", "openai-clip-L-336", "conch_v1_5", "titan"}:
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == "UNI":
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == "GigaPath":
        conf.D_feat = 1536
        conf.D_inner = 768
    else:
        raise ValueError(f"Unsupported pretrain: {conf.pretrain}")


def build_loaders(conf):
    train_data, val_data, test_data = dual_npy_feat_dataset(conf)
    loader_kwargs = {
        "batch_size": conf.B,
        "num_workers": conf.n_worker,
        "pin_memory": conf.pin_memory,
        "collate_fn": collate_patient_dual_list,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def main():
    args = get_arguments()

    with open(args.config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    config.update(vars(args))
    conf = Struct(**config)
    set_feature_dims(conf)

    wandb.init(
        project="wsi_classification",
        config={"dataset": conf.dataset, "pretrain": conf.pretrain, "arch": conf.arch, "seed": conf.seed},
        mode=conf.wandb_mode,
        name=f"{conf.dataset}_{conf.arch}_{conf.pretrain}_seed{conf.seed}",
    )
    ckpt_dir = os.path.join(wandb.run.dir, conf.arch)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")
    print("Used config:")
    pprint(vars(conf))

    set_seed(conf.seed)
    train_loader, val_loader, _test_loader = build_loaders(conf)

    net = DualViewPatientClassifier(conf).to(device)
    criterion = engine.Loss_sum_v1_risk_align_weights()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=conf.lr, weight_decay=conf.wd)

    total_steps = len(train_loader) * conf.train_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    patience = 8
    counter = 0
    best_state = {"fold": conf.seed, "epoch": -1, "val_acc": 0, "val_auc": 0, "val_f1": 0}

    for epoch in range(conf.train_epoch):
        train_one_epoch(net, criterion, train_loader, optimizer, device, epoch, conf)
        scheduler.step()

        val_auc, val_acc, val_f1, val_loss, _g_stats = dual_view_evaluate(net, criterion, val_loader, device, conf, "Val")

        if conf.wandb_mode != "disabled":
            wandb.log({
                "val/val_acc1": val_acc,
                "val/val_auc": val_auc,
                "val/val_f1": val_f1,
                "val/val_loss": val_loss,
            })

        if val_auc * 100 + val_acc > best_state["val_auc"] * 100 + best_state["val_acc"]:
            best_state.update({"epoch": epoch, "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1})
            counter = 0
            save_model(
                conf=conf,
                model=net,
                optimizer=optimizer,
                epoch=epoch,
                save_path=os.path.join(ckpt_dir, f"{conf.dataset}_{conf.arch}_{conf.pretrain}_seed{conf.seed}-best.pth"),
            )
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} with AUC {val_auc:.4f} (No improvement for {patience} epochs)")
            break
        print("\n")

    save_model(
        conf=conf,
        model=net,
        optimizer=optimizer,
        epoch=epoch,
        save_path=os.path.join(ckpt_dir, f"{conf.dataset}_{conf.arch}_{conf.pretrain}_seed{conf.seed}-last.pth"),
    )
    print("Results on best epoch:")
    print(best_state)
    save_results(best_state, args.results_path, conf)
    wandb.finish()


def save_results(best_state, results_path, conf):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    file_exists = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    row_data = {
        "Dataset": conf.dataset,
        "Classes": conf.n_class,
        "Epochs": conf.train_epoch,
        "Seed": conf.seed,
        "Arch": conf.arch,
        "Pretrain": conf.pretrain,
        "LR": conf.lr,
        "Fold": best_state["fold"],
        "Best_Epoch": best_state["epoch"],
        "Acc": f"{best_state['val_acc']:.4f}",
        "AUC": f"{best_state['val_auc']:.4f}",
        "F1": f"{best_state['val_f1']:.4f}",
        "Notes": conf.notes,
    }
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
