#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual NaN-signal MLP for hand->arm mapping (train/eval/resume)
- Inputs: 8 dims = [Lx, Ly, Rx, Ry] (standardized) + [mask_Lx, mask_Ly, mask_Rx, mask_Ry] (1=observed, 0=missing)
- Targets: 4 dims = follower_pos1..4
- Augmentation (train only, optional): with prob p_aug, drop LEFT or RIGHT coords to force masks exactly 0011 or 1100
  * disable with: --augment none  (then original data only)
- Architecture:
    Encoder: 8 -> 16 (proj-res) -> 32 (proj-res) -> ResidualBlock(32)
    Decoder: 32 -> 16 (proj-res) -> 8 (proj-res) -> ResidualBlock(8) -> head 8->4
- Normalization: LayerNorm, GELU
- Resume training: --resume ckpt.pth, with --epochs-add or --epochs
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Default Config ----------------
DEFAULT_CSV = "./unified_log_20250923_204357.csv"
DEFAULT_OUT = "./out_residual_nan_signal_mlp"

IN_VAL_COLS = ["cam_left_x", "cam_left_y", "cam_right_x", "cam_right_y"]
OUT_COLS    = ["follower_pos1", "follower_pos2", "follower_pos3", "follower_pos4"]
MASK_ORDER  = ["cam_left_x","cam_left_y","cam_right_x","cam_right_y"]

# ---------------- Dataset ----------------
class CamNanAugDataset(Dataset):
    def __init__(self, df, in_cols, out_cols, mean_std, split="train",
                 p_aug=0.0, aug_mode="left_right", rng=None):
        """
        aug_mode:
          - 'left_right' : 강제 결측(LEFT 또는 RIGHT)을 p_aug 확률로 적용
          - 'none'       : 증강 완전 비활성화 (원본 NaN 패턴만 사용)
        """
        self.df = df.reset_index(drop=True).copy()
        self.in_cols = in_cols
        self.out_cols = out_cols
        self.mean, self.std = mean_std
        self.split = split
        self.p_aug = float(p_aug) if p_aug is not None else 0.0
        self.aug_mode = (aug_mode or "none").lower()
        self.rng = rng or np.random.default_rng(7)
        self.df = self.df.dropna(subset=out_cols).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vals = row[self.in_cols].values.astype(np.float32)  # [Lx, Ly, Rx, Ry]
        mask = (~np.isnan(vals)).astype(np.float32)

        # (옵션) 증강: train split에서만, aug_mode='left_right'이고 p_aug>0일 때만 수행
        if (self.split == "train" and
            self.aug_mode == "left_right" and
            self.p_aug > 0.0 and
            self.rng.random() < self.p_aug):
            if self.rng.random() < 0.5:
                # Drop LEFT (Lx, Ly)
                vals[0] = np.nan; vals[1] = np.nan
                mask[:] = np.array([0., 0., 1., 1.], dtype=np.float32)
            else:
                # Drop RIGHT (Rx, Ry)
                vals[2] = np.nan; vals[3] = np.nan
                mask[:] = np.array([1., 1., 0., 0.], dtype=np.float32)

        # NaN -> 0 대치 후 표준화
        vals_imp = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        std_safe = np.where(self.std == 0.0, 1.0, self.std)
        vals_std = (vals_imp - self.mean) / std_safe

        # 입력 = 값표준화(4) + 마스크(4)
        x = np.concatenate([vals_std, mask], axis=0).astype(np.float32)
        y = row[self.out_cols].values.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# ---------------- Model ----------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.net(x)

class ProjectedBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        residual = self.proj(x)
        x = self.fc1(x); x = self.act(x); x = self.ln1(x)
        x = self.fc2(x); x = self.act(x); x = self.ln2(x)
        return residual + x

class NaNResidualMLP(nn.Module):
    """
    Input 8 -> 16 -> 32 -> 32 -> 16 -> 8 -> 4
    Encoder residual block at 32 dim, Decoder residual block at 8 dim.
    """
    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.enc1 = ProjectedBlock(in_dim, 16)
        self.enc2 = ProjectedBlock(16, 32)
        self.enc_res = ResidualBlock(32)
        self.dec1 = ProjectedBlock(32, 16)
        self.dec2 = ProjectedBlock(16, 8)
        self.dec_res = ResidualBlock(8)
        self.head = nn.Linear(8, out_dim)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc_res(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec_res(x)
        return self.head(x)

# ---------------- Utils ----------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_split_indices(N, val_ratio, test_ratio, seed=7):
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def fit_input_stats(df_train, in_cols):
    means, stds = [], []
    for c in in_cols:
        col = df_train[c].values.astype(np.float32)
        m = np.nanmean(col); s = np.nanstd(col)
        if np.isnan(m): m = 0.0
        if np.isnan(s) or s == 0.0: s = 1.0
        means.append(m); stds.append(s)
    return np.array(means, np.float32), np.array(stds, np.float32)

def eval_model(model, loader, device, out_cols):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            ys.append(yb.cpu()); ps.append(pred.cpu())
    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()

    mae_raw = mean_absolute_error(y, p, multioutput='raw_values')
    mse_raw = mean_squared_error(y, p, multioutput='raw_values')
    rmse_raw = np.sqrt(mse_raw)
    r2_raw = np.array([r2_score(y[:, i], p[:, i]) for i in range(y.shape[1])], dtype=np.float32)

    rows = []
    for i, name in enumerate(out_cols):
        rows.append({
            "dim":  name,
            "MAE":  float(mae_raw[i]),
            "MSE":  float(mse_raw[i]),
            "RMSE": float(rmse_raw[i]),
            "R2":   float(r2_raw[i]),
        })

    summary = {
        "MAE_mean":  float(np.mean(mae_raw)),
        "MSE_mean":  float(np.mean(mse_raw)),
        "RMSE_mean": float(np.mean(rmse_raw)),
        "R2_mean":   float(np.mean(r2_raw)),
    }
    return y, p, rows, summary

def print_split(split_name, summary, rows):
    print(f"\n=== {split_name.upper()} METRICS ===")
    print(f"Avg  -> MAE: {summary['MAE_mean']:.4f} | MSE: {summary['MSE_mean']:.4f} "
          f"| RMSE: {summary['RMSE_mean']:.4f} | R2: {summary['R2_mean']:.4f}")
    for r in rows:
        print(f" - {r['dim']:>12s} | MAE: {r['MAE']:.4f} | MSE: {r['MSE']:.4f} "
              f"| RMSE: {r['RMSE']:.4f} | R2: {r['R2']:.4f}")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Train Residual NaN-signal MLP (resume supported)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Training CSV path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output directory")
    ap.add_argument("--epochs", type=int, default=2000, help="Total epochs (or target total if --resume)")
    ap.add_argument("--epochs-add", type=int, default=None, help="Additional epochs when resuming")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)

    # 🔽 증강 제어 파라미터: none으로 끄면 원본 데이터만 사용
    ap.add_argument("--augment", choices=["left_right", "none"], default="left_right",
                    help="left_right: 강제 결측 증강 / none: 증강 비활성화")
    ap.add_argument("--p-aug", type=float, default=0.30,
                    help="강제 결측 증강 확률 (augment=left_right일 때만 사용)")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--small-epochs", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume from")
    ap.add_argument("--warm-start", action="store_true", help="Load weights only (ignore optimizer state)")
    ap.add_argument("--resume-lr", type=float, default=None, help="Override LR when resuming")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="Gradient clipping (0=off)")
    ap.add_argument("--no-save-plots", action="store_true", help="Skip saving scatter plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Augmentation mode: {args.augment} (p_aug={args.p_aug if args.augment=='left_right' else 0.0})")

    # Load CSV
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)

    # Column checks
    for c in IN_VAL_COLS + OUT_COLS:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")

    df_nonan_targets = df.dropna(subset=OUT_COLS).reset_index(drop=True)

    # Optional subsample
    if args.subsample is not None and len(df_nonan_targets) > args.subsample:
        rng = np.random.default_rng(args.seed)
        ridx = rng.choice(len(df_nonan_targets), size=args.subsample, replace=False)
        df_nonan_targets = df_nonan_targets.iloc[ridx].reset_index(drop=True)

    N = len(df_nonan_targets)
    train_idx, val_idx, test_idx = compute_split_indices(N, args.val_ratio, args.test_ratio, seed=args.seed)
    df_train = df_nonan_targets.iloc[train_idx].copy()
    df_val   = df_nonan_targets.iloc[val_idx].copy()
    df_test  = df_nonan_targets.iloc[test_idx].copy()

    # Stats from train
    means, stds = fit_input_stats(df_train, IN_VAL_COLS)

    # Build model/opt
    model = NaNResidualMLP(in_dim=8, out_dim=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()

    # Resume
    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume file not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        if "state_dict" not in ckpt:
            raise KeyError("Checkpoint missing 'state_dict'")
        model.load_state_dict(ckpt["state_dict"])
        print(f"[Resume] Loaded weights from {args.resume}")

        # means/stds
        if "means" in ckpt and "stds" in ckpt:
            means = np.array(ckpt["means"], dtype=np.float32)
            stds  = np.array(ckpt["stds"],  dtype=np.float32)
            print("[Resume] Using means/stds from checkpoint")

        # optimizer
        if not args.warm_start and "optimizer_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
                print("[Resume] Optimizer state loaded")
            except Exception as e:
                print(f"[Resume] Optimizer state load failed ({e}); continuing without it)")

        # LR override
        if args.resume_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(args.resume_lr)
            print(f"[Resume] LR overridden to {args.resume_lr}")

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        # (참고) augment/p_aug는 재학습 시 새로 주는 CLI 인자 우선 적용
        print(f"[Resume] start_epoch={start_epoch}, best_val={best_val:.6f}")

    # Datasets / loaders
    rng_ds = np.random.default_rng(args.seed)
    p_aug_eff = args.p_aug if args.augment == "left_right" else 0.0

    train_ds = CamNanAugDataset(df_train, IN_VAL_COLS, OUT_COLS, (means, stds),
                                split="train", p_aug=p_aug_eff, aug_mode=args.augment, rng=rng_ds)
    val_ds   = CamNanAugDataset(df_val,   IN_VAL_COLS, OUT_COLS, (means, stds),
                                split="val",   p_aug=0.0, aug_mode="none")
    test_ds  = CamNanAugDataset(df_test,  IN_VAL_COLS, OUT_COLS, (means, stds),
                                split="test",  p_aug=0.0, aug_mode="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Epoch schedule
    if args.small_epochs is not None:
        total_epochs = start_epoch + int(args.small_epochs)
        print(f"[SMOKE] Overriding epochs -> will train {args.small_epochs} more (to epoch {total_epochs})")
    elif args.epochs_add is not None:
        total_epochs = start_epoch + int(args.epochs_add)
        print(f"[Resume] epochs_add={args.epochs_add} -> training to epoch {total_epochs}")
    else:
        total_epochs = int(args.epochs)
        if total_epochs <= start_epoch:
            print(f"[Resume] Target total epochs ({total_epochs}) <= start_epoch ({start_epoch}). Nothing to do.")

    # Train loop
    history_rows = []
    for ep in range(start_epoch + 1, total_epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            opt.step()

            tr_loss_sum += loss.item() * xb.size(0)
            tr_count += xb.size(0)

        tr_loss = tr_loss_sum / max(tr_count, 1)

        # validation
        model.eval()
        vl_loss_sum = 0.0
        vl_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                l = loss_fn(model(xb), yb)
                vl_loss_sum += l.item() * xb.size(0)
                vl_count += xb.size(0)
        vl_loss = vl_loss_sum / max(vl_count, 1)

        print(f"[{ep}] train={tr_loss:.6f}  val={vl_loss:.6f}")

        # Save "last" checkpoint each epoch
        last_path = os.path.join(args.out, "last.pth")
        torch.save({
            "state_dict": model.state_dict(),
            "in_cols_values": IN_VAL_COLS,
            "mask_order": MASK_ORDER,
            "out_cols": OUT_COLS,
            "means": means,
            "stds": stds,
            "augment": args.augment,
            "p_aug": p_aug_eff,
            "optimizer_state": opt.state_dict(),
            "epoch": ep,
            "best_val": best_val,
        }, last_path)

        # Track best on val (by MAE/L1)
        if vl_loss < best_val:
            best_val = vl_loss
            best_path = os.path.join(args.out, "residual_nan_signal_mlp.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "in_cols_values": IN_VAL_COLS,
                "mask_order": MASK_ORDER,
                "out_cols": OUT_COLS,
                "means": means,
                "stds": stds,
                "augment": args.augment,
                "p_aug": p_aug_eff,
                "optimizer_state": opt.state_dict(),
                "epoch": ep,
                "best_val": best_val,
            }, best_path)
            print(f"  ✅ Saved BEST to {best_path} (val={best_val:.6f})")

        history_rows.append({"epoch": ep, "train_loss": tr_loss, "val_loss": vl_loss})

    # Reload best before final eval
    best_path = os.path.join(args.out, "residual_nan_signal_mlp.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[Eval] Loaded best checkpoint @ epoch {ckpt.get('epoch')} (val={ckpt.get('best_val'):.6f})")

    # Evaluate
    y_tr, p_tr, rows_tr, sum_tr = eval_model(model, train_loader, device, OUT_COLS)
    y_vl, p_vl, rows_vl, sum_vl = eval_model(model, val_loader,   device, OUT_COLS)
    y_ts, p_ts, rows_ts, sum_ts = eval_model(model, test_loader,  device, OUT_COLS)

    print_split("train", sum_tr, rows_tr)
    print_split("val",   sum_vl, rows_vl)
    print_split("test",  sum_ts, rows_ts)

    # Save metrics + history + plots
    rows = []
    for r in rows_tr: rows.append({"split":"train", **r})
    for r in rows_vl: rows.append({"split":"val",   **r})
    for r in rows_ts: rows.append({"split":"test",  **r})
    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(args.out, "metrics_residual.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(args.out, "train_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    if not args.no_save_plots:
        for i, name in enumerate(OUT_COLS):
            save_png = os.path.join(args.out, f"preds_vs_true_{name}.png")
            plt.figure()
            plt.scatter(y_ts[:, i], p_ts[:, i], s=6)
            mn = min(np.min(y_ts[:, i]), np.min(p_ts[:, i]))
            mx = max(np.max(y_ts[:, i]), np.max(p_ts[:, i]))
            plt.plot([mn, mx], [mn, mx])
            plt.xlabel("True"); plt.ylabel("Predicted")
            plt.title(f"Test Pred vs True ({name})")
            plt.tight_layout(); plt.savefig(save_png, dpi=140); plt.close()

    print("\nSaved files:")
    print(f"- Best checkpoint: {best_path if os.path.exists(best_path) else '(not improved)'}")
    print(f"- Last checkpoint: {os.path.join(args.out, 'last.pth')}")
    print(f"- Metrics CSV:     {metrics_csv}")
    print(f"- Train history:   {hist_csv}")
    if not args.no_save_plots:
        for name in OUT_COLS:
            print(f"- Plot:            {os.path.join(args.out, f'preds_vs_true_{name}.png')}")

if __name__ == "__main__":
    main()
