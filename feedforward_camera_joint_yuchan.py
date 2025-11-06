#!/usr/bin/env python3
"""
Simple Feedforward Model for Camera to Joint Mapping
Adds: --init-from to load pretrained weights (resume/fine-tune)
"""

import argparse
from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class CameraJointDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleFeedforward(nn.Module):
    """Configurable feedforward regression network."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 5,
        d_model: int = 8,            # not used directly (kept for CLI compatibility)
        nhead: int = 1,              # not used (kept for CLI compatibility)
        num_layers: int = 1,
        dim_feedforward: int = 12,   # used if hidden_sizes is None
        dropout: float = 0.0,
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            if num_layers <= 0:
                raise ValueError("num_layers must be positive when hidden_sizes is not provided")
            hidden_sizes = tuple(dim_feedforward for _ in range(num_layers)) or (dim_feedforward,)

        hidden_sizes_tuple = tuple(hidden_sizes)
        if not hidden_sizes_tuple:
            raise ValueError("hidden_sizes must contain at least one layer")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes_tuple:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FeedforwardCameraJointRegressor:
    """Feedforward neural network regressor for camera to joint mapping"""

    def __init__(
        self,
        d_model: int = 8,
        nhead: int = 1,
        num_layers: int = 1,
        dim_feedforward: int = 12,
        hidden_sizes: Sequence[int] | None = (8, 16, 8),
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        epochs: int = 800,
        device: torch.device | None = None,
        init_from: str | None = None,
        strict_load: bool = False,
        reuse_scaler: bool = False,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.hidden_sizes = tuple(hidden_sizes) if hidden_sizes is not None else None
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"🔧 Using device: {self.device}")

        # Scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.reuse_scaler = reuse_scaler

        # Training history
        self.train_losses = []
        self.val_losses = []

        # Pretrained checkpoint (optional)
        self.init_from = init_from
        self.strict_load = strict_load
        self._ckpt_loaded = False
        self._ckpt_model_config = None
        self._ckpt_state_dict = None
        self._maybe_load_checkpoint_meta()

    # ---------- Checkpoint load helpers ----------
    def _maybe_load_checkpoint_meta(self):
        """If init_from is provided, read checkpoint and cache contents.
        Supports:
          - full dict with 'model_state_dict' (+ 'model_config', 'scaler_X', 'scaler_y', etc.)
          - bare state_dict (tensor name -> weight)
        """
        if self.init_from is None:
            return
        print(f"📦 Loading checkpoint meta from: {self.init_from}")
        obj = torch.load(self.init_from, map_location="cpu")

        # Full package?
        if isinstance(obj, dict) and "model_state_dict" in obj:
            self._ckpt_state_dict = obj["model_state_dict"]
            self._ckpt_model_config = obj.get("model_config", None)
            # Optionally reuse scaler
            if self.reuse_scaler:
                if "scaler_X" in obj and "scaler_y" in obj:
                    self.scaler_X = obj["scaler_X"]
                    self.scaler_y = obj["scaler_y"]
                    print("🔁 Reusing scalers from checkpoint.")
                else:
                    print("⚠️ Checkpoint has no scalers; reuse_scaler ignored.")
            print("✅ Checkpoint (full package) meta read.")
            self._ckpt_loaded = True
        else:
            # Assume bare state_dict
            if isinstance(obj, dict):
                self._ckpt_state_dict = obj
                print("✅ Checkpoint (bare state_dict) meta read.")
                self._ckpt_loaded = True
            else:
                raise ValueError("Unsupported checkpoint format.")

    def _build_model_with_ckpt_if_available(self):
        """Create model. If a checkpoint provides model_config and user didn't
        explicitly override architecture, use the ckpt architecture."""
        use_ckpt_arch = False
        ck = self._ckpt_model_config

        # Detect if user explicitly set hidden_sizes/num_layers/dim_feedforward
        user_overrode_arch = self.hidden_sizes is not None or (self.num_layers is not None)

        if ck is not None and not user_overrode_arch:
            # adopt checkpoint architecture
            self.d_model = ck.get("d_model", self.d_model)
            self.nhead = ck.get("nhead", self.nhead)
            self.num_layers = ck.get("num_layers", self.num_layers)
            self.dim_feedforward = ck.get("dim_feedforward", self.dim_feedforward)
            self.hidden_sizes = tuple(ck.get("hidden_sizes", self.hidden_sizes or (self.dim_feedforward,)))
            self.dropout = ck.get("dropout", self.dropout)
            use_ckpt_arch = True

        if use_ckpt_arch:
            print(f"🧩 Using checkpoint architecture: hidden_sizes={self.hidden_sizes}")
        else:
            print(f"🧩 Using user-specified architecture: hidden_sizes={self.hidden_sizes or (self.dim_feedforward,)*self.num_layers}")

        model = SimpleFeedforward(
            input_dim=4,
            output_dim=5,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            hidden_sizes=self.hidden_sizes
        ).to(self.device)

        # Load weights if present
        if self._ckpt_state_dict is not None:
            missing, unexpected = model.load_state_dict(self._ckpt_state_dict, strict=self.strict_load)
            if not self.strict_load:
                # torch<2 returns tuple of lists; torch>=2 returns IncompatibleKeys
                if isinstance(missing, list) and isinstance(unexpected, list):
                    if missing:
                        print(f"ℹ️ Missing keys (ok with strict=False): {missing[:6]}{' ...' if len(missing)>6 else ''}")
                    if unexpected:
                        print(f"ℹ️ Unexpected keys (ok with strict=False): {unexpected[:6]}{' ...' if len(unexpected)>6 else ''}")
                else:
                    # torch>=2 path
                    if missing.missing_keys:
                        print(f"ℹ️ Missing keys (ok with strict=False): {missing.missing_keys[:6]}{' ...' if len(missing.missing_keys)>6 else ''}")
                    if missing.unexpected_keys:
                        print(f"ℹ️ Unexpected keys (ok with strict=False): {missing.unexpected_keys[:6]}{' ...' if len(missing.unexpected_keys)>6 else ''}")
            else:
                print("✅ Weights loaded with strict=True.")

        return model

    # ---------- Data pipeline ----------
    def load_data(self, csv_file='unified_log_20250922_192827.csv'):
        print(f"📁 Loading data from {csv_file}")
        self.raw_data = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(self.raw_data)} rows")
        return self.raw_data

    def preprocess_data(self):
        print("🔧 Preprocessing data...")
        feature_cols = ['cam_left_x', 'cam_left_y', 'cam_right_x', 'cam_right_y']
        target_cols = ['follower_pos1', 'follower_pos2', 'follower_pos3', 'follower_pos4', 'follower_pos5']

        data_clean = self.raw_data.dropna(subset=feature_cols + target_cols)
        print(f"📊 After removing NaN values: {len(data_clean)} rows ({len(self.raw_data) - len(data_clean)} removed)")
        if len(data_clean) == 0:
            raise ValueError("No valid data remaining after preprocessing")

        X = data_clean[feature_cols].values
        y = data_clean[target_cols].values

        X_valid = np.isfinite(X).all(axis=1)
        y_valid = np.isfinite(y).all(axis=1)
        valid_mask = X_valid & y_valid
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📊 Final dataset: {len(X)} samples with {X.shape[1]} features -> {y.shape[1]} targets")
        print("📈 Feature ranges:")
        for i, col in enumerate(feature_cols):
            print(f"  {col}: {X[:, i].min():.1f} - {X[:, i].max():.1f}")
        print("📈 Target ranges:")
        for i, col in enumerate(target_cols):
            print(f"  {col}: {y[:, i].min():.1f} - {y[:, i].max():.1f}")

        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        return X, y

    def split_data(self, test_size=0.0, random_state=42):
        if test_size == 0.0:
            self.X_train = self.X
            self.X_test = self.X
            self.y_train = self.y
            self.y_test = self.y
            print(f"📊 Using all {len(self.X)} samples for training (maximum overfitting)")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            print(f"📊 Train set: {len(self.X_train)} samples")
            print(f"📊 Test set: {len(self.X_test)} samples")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ---------- Train / Eval ----------
    def train(self, normalize=True):
        print("🤖 Training simplified neural network...")

        # scaler policy
        if normalize:
            if self.reuse_scaler and self._ckpt_loaded:
                # Reusing scalers already set in _maybe_load_checkpoint_meta()
                X_train_scaled = self.scaler_X.transform(self.X_train)
                y_train_scaled = self.scaler_y.transform(self.y_train)
                X_val_scaled = self.scaler_X.transform(self.X_test)
                y_val_scaled = self.scaler_y.transform(self.y_test)
                print("🔁 Using checkpoint scalers for normalization.")
            else:
                X_train_scaled = self.scaler_X.fit_transform(self.X_train)
                y_train_scaled = self.scaler_y.fit_transform(self.y_train)
                X_val_scaled = self.scaler_X.transform(self.X_test)
                y_val_scaled = self.scaler_y.transform(self.y_test)
        else:
            X_train_scaled = self.X_train
            y_train_scaled = self.y_train
            X_val_scaled = self.X_test
            y_val_scaled = self.y_test

        self.normalize = normalize

        train_dataset = CameraJointDataset(X_train_scaled, y_train_scaled)
        val_dataset = CameraJointDataset(X_val_scaled, y_val_scaled)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Build (and maybe load) model
        self.model = self._build_model_with_ckpt_if_available()

        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.hidden_sizes is not None:
            print(f"   Hidden sizes: {self.hidden_sizes}")

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predictions = self.model(batch_features)
                    loss = criterion(predictions, batch_targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_feedforward_model.pth')

            if epoch % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        self.model.load_state_dict(torch.load('best_feedforward_model.pth', map_location=self.device))
        print("✅ Training completed")

    def predict(self, X):
        self.model.eval()
        if self.normalize:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        if self.normalize:
            predictions = self.scaler_y.inverse_transform(predictions)
        return predictions

    def evaluate(self):
        print("📊 Evaluating feedforward model performance...")
        y_pred = self.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("📈 Overall Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")

        print("📈 Per-Joint Performance:")
        for i, target_col in enumerate(self.target_cols):
            mse_i = mean_squared_error(self.y_test[:, i], y_pred[:, i])
            rmse_i = np.sqrt(mse_i)
            mae_i = mean_absolute_error(self.y_test[:, i], y_pred[:, i])
            r2_i = r2_score(self.y_test[:, i], y_pred[:, i])
            print(f"  {target_col}:")
            print(f"    RMSE: {rmse_i:.2f}, MAE: {mae_i:.2f}, R²: {r2_i:.4f}")

        self.test_metrics = {
            'rmse': rmse, 'mae': mae, 'r2': r2,
            'y_test': self.y_test, 'y_pred': y_pred
        }
        return self.test_metrics

    def plot_results(self, save_plot=True):
        if not hasattr(self, 'test_metrics'):
            print("❌ No test metrics available. Run evaluate() first.")
            return

        y_test = self.test_metrics['y_test']
        y_pred = self.test_metrics['y_pred']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        from sklearn.metrics import r2_score
        for i, target_col in enumerate(self.target_cols):
            ax = axes[i]
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)
            min_val = min(y_test[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
            ax.set_title(f'{target_col} (R² = {r2_score(y_test[:, i], y_pred[:, i]):.4f})')
            ax.legend(); ax.grid(True, alpha=0.3)

        axes[5].axis('off')
        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"feedforward_results_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_filename}")
        plt.show()

        if self.train_losses and self.val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
            plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Feedforward Training History')
            plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
            if save_plot:
                history_filename = f"feedforward_training_history_{timestamp}.png"
                plt.savefig(history_filename, dpi=300, bbox_inches='tight')
                print(f"📊 Training history saved: {history_filename}")
            plt.show()

    def save_model(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedforward_camera_joint_{timestamp}.pth"
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout
            },
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols,
            'normalize': self.normalize
        }
        torch.save(save_dict, filename)
        print(f"💾 Feedforward model saved: {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description="Train feedforward network for camera to joint mapping")
    parser.add_argument("--data", default="unified_log_20250923_204357.csv", help="CSV file to use")
    parser.add_argument("--test-size", type=float, default=0.0, help="Test set size (default: 0.0 - use all data)")
    parser.add_argument("--no-normalize", action='store_true', help="Disable feature normalization")
    parser.add_argument("--no-plot", action='store_true', help="Skip plotting results")
    parser.add_argument("--output", help="Output model filename")

    # Architecture
    parser.add_argument("--d-model", type=int, default=8, help="Hidden layer dimension (compat only)")
    parser.add_argument("--nhead", type=int, default=1, help="Number of attention heads (compat only)")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden-sizes", default="8, 16, 32, 16, 8",
                        help="Comma-separated hidden layer sizes (overrides --num-layers)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate applied after each hidden layer")

    # Train
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # Fine-tune / resume
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to .pth checkpoint to initialize weights from")
    parser.add_argument("--strict-load", action="store_true",
                        help="Use strict=True when loading checkpoint (default: False)")
    parser.add_argument("--reuse-scaler", action="store_true",
                        help="Reuse scaler_X / scaler_y from checkpoint (default: fit on new data)")

    args = parser.parse_args()

    # Parse hidden_sizes
    hidden_sizes: Sequence[int] | None
    if args.hidden_sizes:
        try:
            hidden_sizes = tuple(int(size.strip()) for size in args.hidden_sizes.split(',') if size.strip())
        except ValueError as exc:
            parser.error(f"Invalid --hidden-sizes value '{args.hidden_sizes}': {exc}")
        if not hidden_sizes:
            hidden_sizes = None
    else:
        hidden_sizes = None

    resolved_num_layers = len(hidden_sizes) if hidden_sizes is not None else args.num_layers

    print("🤖 Feedforward Model Training for Camera to Joint Mapping")
    print("=" * 60)

    try:
        regressor = FeedforwardCameraJointRegressor(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=resolved_num_layers,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            init_from=args.init_from,
            strict_load=args.strict_load,
            reuse_scaler=args.reuse_scaler,
        )

        regressor.load_data(args.data)
        regressor.preprocess_data()
        regressor.split_data(test_size=args.test_size)

        regressor.train(normalize=not args.no_normalize)
        regressor.evaluate()

        if not args.no_plot:
            regressor.plot_results()

        regressor.save_model(args.output)
        print("✅ Feedforward training completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
