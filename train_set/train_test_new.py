#!/usr/bin/env python3
"""
ResNet-style Feedforward Model for Camera to Joint Mapping
- Architecture: 4 -> 8 -> 16 (Block A, residual) -> 32 -> 16 -> 8 (Block B, residual) -> 5
- Long skip: original 4-dim input -> final 5-dim output
- CLI와 내부 CONFIG 동시 지원: 기본은 CLI가 우선, --use-internal 시 CONFIG 강제 적용
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 내부 기본 설정(원하면 여기만 수정)
# =========================
CONFIG = {
    "data": r"./unified_log_20250923_204357.csv",
    "test_size": 0.0,          # 0.0이면 전부 학습
    "normalize": True,         # True면 StandardScaler 사용
    "plot": True,              # True면 플롯 저장/표시
    "output": None,            # None이면 타임스탬프 자동 파일명
    "epochs": 2000,
    "lr": 1e-3,
    "batch_size": 1,
    "dropout": 0.0,            # ResBlock dropout
}

# =========================
# 데이터셋
# =========================
class CameraJointDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# =========================
# ResNet 스타일 블록
# =========================
class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        out = self.drop(out)
        out = out + self.proj(x)
        out = self.act(self.norm(out))
        return out

class ResFeedforward(nn.Module):
    """
    변경 사항:
      - 입력 4 → 선형 8 → BlockA(8,16,32) → BlockB(32,16,8) → 선형 5
      - Long skip: Linear(4 -> 5)
    노드 개수: 8, 16, 32, 32, 16, 8  (히든 너비 기준)
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        # 4 -> 8
        self.fc_in = nn.Linear(4, 8)

        # Block A: 8 -> 16 -> 32 (residual)
        self.block_a = ResBlock(8, 16, 32, dropout=dropout)

        # Block B: 32 -> 16 -> 8 (residual)
        self.block_b = ResBlock(32, 16, 8, dropout=dropout)

        # 8 -> 5 (최종 출력 5개 모터)
        self.fc_out = nn.Linear(8, 5)

        # Long skip: 입력 4 -> 출력 5
        self.long_skip = nn.Linear(4, 5)

        # init
        for m in (self.fc_in, self.fc_out, self.long_skip):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))   # 4 -> 8
        h = self.block_a(h)           # 8 -> 32
        h = self.block_b(h)           # 32 -> 8
        y = self.fc_out(h)            # 8 -> 5
        y = y + self.long_skip(x)     # 4 -> 5 (long skip)
        return y

# =========================
# Regressor
# =========================
class FeedforwardCameraJointRegressor:
    def __init__(self, learning_rate=1e-3, batch_size=8, epochs=800, device=None, dropout=0.0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        print(f"🔧 Using device: {self.device}")

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.train_losses = []
        self.val_losses = []

    def load_data(self, csv_file):
        print(f"📁 Loading data from {csv_file}")
        self.raw_data = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(self.raw_data)} rows")
        return self.raw_data

    def preprocess_data(self):
        print("🔧 Preprocessing data...")
        feature_cols = ['cam_left_x', 'cam_left_y', 'cam_right_x', 'cam_right_y']
        # ★ 출력 5개로 변경
        target_cols = ['follower_pos1', 'follower_pos2', 'follower_pos3', 'follower_pos4', 'follower_pos5']

        data_clean = self.raw_data.dropna(subset=feature_cols + target_cols)
        print(f"📊 After removing NaN: {len(data_clean)} rows ({len(self.raw_data) - len(data_clean)} removed)")
        if len(data_clean) == 0:
            raise ValueError("No valid data after preprocessing")

        X = data_clean[feature_cols].values
        y = data_clean[target_cols].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        X, y = X[mask], y[mask]

        print(f"📊 Final dataset: {len(X)} samples | X:{X.shape} → y:{y.shape}")
        self.X, self.y = X, y
        self.feature_cols, self.target_cols = feature_cols, target_cols
        return X, y

    def split_data(self, test_size=0.0, random_state=42):
        if test_size == 0.0:
            self.X_train = self.X
            self.X_test = self.X
            self.y_train = self.y
            self.y_test = self.y
            print(f"📊 Using ALL {len(self.X)} samples for training (max overfit)")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            print(f"📊 Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, normalize=True):
        print("🤖 Training ResNet-style model...")
        if normalize:
            X_train = self.scaler_X.fit_transform(self.X_train)
            y_train = self.scaler_y.fit_transform(self.y_train)
            X_val = self.scaler_X.transform(self.X_test)
            y_val = self.scaler_y.transform(self.y_test)
        else:
            X_train, y_train = self.X_train, self.y_train
            X_val, y_val = self.X_test, self.y_test

        self.normalize = normalize

        train_loader = DataLoader(CameraJointDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(CameraJointDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        # ★ 새 너비/출력 반영된 모델
        self.model = ResFeedforward(dropout=self.dropout).to(self.device)
        print(f"🧠 Params: {sum(p.numel() for p in self.model.parameters()):,}")

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)
        criterion = nn.MSELoss()

        best_val = float('inf')
        patience = 200
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            tr_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= max(1, len(train_loader))
            self.train_losses.append(tr_loss)

            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    va_loss += criterion(pred, yb).item()
            va_loss /= max(1, len(val_loader))
            self.val_losses.append(va_loss)

            scheduler.step(va_loss)

            if va_loss < best_val:
                best_val = va_loss
                wait = 0
                torch.save(self.model.state_dict(), 'best_feedforward_model.pth')
            else:
                wait += 1
                if wait >= patience:
                    print(f"⏹ Early stopping @ epoch {epoch}")
                    break

            if epoch % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | train {tr_loss:.6f} | val {va_loss:.6f} | lr {lr:.6f}")

        self.model.load_state_dict(torch.load('best_feedforward_model.pth'))
        print("✅ Training finished")

    def predict(self, X):
        self.model.eval()
        Xs = self.scaler_X.transform(X) if self.normalize else X
        Xt = torch.FloatTensor(Xs).to(self.device)
        with torch.no_grad():
            pred = self.model(Xt).cpu().numpy()
        return self.scaler_y.inverse_transform(pred) if self.normalize else pred

    def evaluate(self):
        print("📊 Evaluating...")
        y_pred = self.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Overall | RMSE {rmse:.3f} | MAE {mae:.3f} | R² {r2:.4f}")
        print("Per-joint:")
        for i, name in enumerate(self.target_cols):
            rmse_i = np.sqrt(mean_squared_error(self.y_test[:, i], y_pred[:, i]))
            mae_i  = mean_absolute_error(self.y_test[:, i], y_pred[:, i])
            r2_i   = r2_score(self.y_test[:, i], y_pred[:, i])
            print(f"  {name:14s} RMSE {rmse_i:.3f} | MAE {mae_i:.3f} | R² {r2_i:.4f}")

        self.test_metrics = {"rmse": rmse, "mae": mae, "r2": r2, "y_test": self.y_test, "y_pred": y_pred}
        return self.test_metrics

    def plot_results(self, save_plot=True):
        if not hasattr(self, 'test_metrics'):
            print("❌ Run evaluate() first.")
            return
        y_test = self.test_metrics['y_test']
        y_pred = self.test_metrics['y_pred']

        # Scatter: each joint
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        # 주의: 현재 5개 출력이므로 플롯은 4개만 기본 표시 (기존 로직 유지)
        # 필요시 여기 확장 가능하지만, "다른건 고치지 말라" 요청에 따라 유지
        for i, name in enumerate(self.target_cols[:4]):
            ax = axes[i]
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)
            lo, hi = min(y_test[:, i].min(), y_pred[:, i].min()), max(y_test[:, i].max(), y_pred[:, i].max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name} (R²={r2_score(y_test[:, i], y_pred[:, i]):.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_plot:
            fn = f"feedforward_results_{ts}.png"
            plt.savefig(fn, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {fn}")
        plt.show()

        # History
        if self.train_losses and self.val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Train', alpha=0.7)
            plt.plot(self.val_losses, label='Val', alpha=0.7)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
            plt.title('Training History'); plt.legend(); plt.grid(True, alpha=0.3)
            if save_plot:
                fn = f"feedforward_training_history_{ts}.png"
                plt.savefig(fn, dpi=300, bbox_inches='tight')
                print(f"📊 History saved: {fn}")
            plt.show()

    def save_model(self, filename=None):
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedforward_camera_joint_{ts}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {"arch": "ResFF(4→8→[8-16-32]→[32-16-8]→5)+long-skip(4→5)", "dropout": self.dropout},
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "normalize": self.normalize
        }, filename)
        print(f"💾 Saved: {filename}")
        return filename

# =========================
# main
# =========================
def build_parser():
    p = argparse.ArgumentParser(description="Train ResNet-style feedforward model (camera→joint)")
    p.add_argument("--data", default=CONFIG["data"])
    p.add_argument("--test-size", type=float, default=CONFIG["test_size"])
    p.add_argument("--no-normalize", action='store_true')
    p.add_argument("--no-plot", action='store_true')
    p.add_argument("--output", default=CONFIG["output"])
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=CONFIG["lr"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    p.add_argument("--use-internal", action='store_true',
                   help="Ignore CLI and use CONFIG values as-is")
    return p

def resolve_args(args):
    """
    CONFIG와 CLI를 병합.
    - 기본: CLI 값 우선
    - --use-internal 지정 시: CONFIG 값 강제 사용
    """
    if args.use_internal:
        merged = CONFIG.copy()
    else:
        merged = {
            "data": args.data,
            "test_size": args.test_size,
            "normalize": not args.no_normalize,
            "plot": not args.no_plot,
            "output": args.output,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
        }
        # CONFIG에만 있고 CLI에 없는 키가 생기면 보존
        for k, v in CONFIG.items():
            merged.setdefault(k, v)
    return merged

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_args(args)

    print("🤖 Training (ResNet blocks)")
    print("=" * 60)
    print("Final Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    try:
        reg = FeedforwardCameraJointRegressor(
            learning_rate=cfg["lr"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            dropout=cfg["dropout"]
        )
        reg.load_data(cfg["data"])
        reg.preprocess_data()
        reg.split_data(test_size=cfg["test_size"])
        reg.train(normalize=cfg["normalize"])
        reg.evaluate()
        if cfg["plot"]:
            reg.plot_results(save_plot=True)
        reg.save_model(cfg["output"])
        print("✅ Done!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
