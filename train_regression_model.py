#!/usr/bin/env python3
"""
Regression Model Training for Camera to Joint Mapping
Trains a model to predict 4 joint positions from 4 camera coordinates
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import argparse
from datetime import datetime

class ResidualMLPRegressor:
    """Custom MLP with residual connections using numpy"""

    def __init__(self, learning_rate=0.001, epochs=1000, batch_size=32,
                 dropout_rate=0.2, weight_decay=0.0001, early_stopping_patience=50):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.weights = {}
        self.biases = {}

        # For learning rate scheduling
        self.lr_decay_factor = 0.9
        self.lr_decay_patience = 20

        # Remove batch normalization variables

        # Training history
        self.train_losses = []
        self.val_losses = []

    def _initialize_weights(self, input_dim):
        """Initialize weights for maximum overfitting network"""
        # Much larger network: 4 -> 64 -> 128 -> 64 -> 32 -> 4
        self.weights['W1'] = np.random.randn(input_dim, 64) * np.sqrt(2.0 / input_dim)
        self.biases['b1'] = np.zeros((1, 64))

        self.weights['W2'] = np.random.randn(64, 128) * np.sqrt(2.0 / 64)
        self.biases['b2'] = np.zeros((1, 128))

        self.weights['W3'] = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.biases['b3'] = np.zeros((1, 64))

        self.weights['W4'] = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.biases['b4'] = np.zeros((1, 32))

        self.weights['W5'] = np.random.randn(32, 4) * np.sqrt(2.0 / 32)
        self.biases['b5'] = np.zeros((1, 4))

        # Multiple residual connections
        self.weights['Wr1'] = np.random.randn(64, 64) * np.sqrt(2.0 / 64)  # 64 -> 64 skip
        self.weights['Wr2'] = np.random.randn(64, 32) * np.sqrt(2.0 / 64)  # 64 -> 32 skip

        # Remove batch normalization initialization
        pass

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def _leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def _dropout(self, x, training=True):
        if not training:
            return x
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * mask

    def _batch_norm(self, x, layer, training=True, epsilon=1e-8):
        if training:
            # Training mode: compute batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Update running statistics
            self.bn_running_mean[layer] = (self.bn_momentum * self.bn_running_mean[layer] +
                                         (1 - self.bn_momentum) * batch_mean)
            self.bn_running_var[layer] = (self.bn_momentum * self.bn_running_var[layer] +
                                        (1 - self.bn_momentum) * batch_var)

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + epsilon)
        else:
            # Inference mode: use running statistics
            x_norm = (x - self.bn_running_mean[layer]) / np.sqrt(self.bn_running_var[layer] + epsilon)

        # Scale and shift
        return self.bn_gamma[layer] * x_norm + self.bn_beta[layer]

    def forward(self, X, training=True):
        """Forward pass with maximum overfitting capacity"""
        # Layer 1: 4 -> 64
        z1 = np.dot(X, self.weights['W1']) + self.biases['b1']
        a1 = self._leaky_relu(z1)  # 64 dim
        a1 = self._dropout(a1, training)

        # Layer 2: 64 -> 128
        z2 = np.dot(a1, self.weights['W2']) + self.biases['b2']
        a2 = self._leaky_relu(z2)  # 128 dim
        a2 = self._dropout(a2, training)

        # Layer 3: 128 -> 64 with residual from a1
        z3 = np.dot(a2, self.weights['W3']) + self.biases['b3']
        skip1 = np.dot(a1, self.weights['Wr1'])  # 64 -> 64 projection
        a3 = self._leaky_relu(z3 + skip1)  # 64 dim with residual
        a3 = self._dropout(a3, training)

        # Layer 4: 64 -> 32 with residual from a1
        z4 = np.dot(a3, self.weights['W4']) + self.biases['b4']
        skip2 = np.dot(a1, self.weights['Wr2'])  # 64 -> 32 projection
        a4 = self._leaky_relu(z4 + skip2)  # 32 dim with residual
        a4 = self._dropout(a4, training)

        # Output layer: 32 -> 4
        z5 = np.dot(a4, self.weights['W5']) + self.biases['b5']
        output = z5  # No activation on output layer

        return output, {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4,
                       'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5,
                       'skip1': skip1, 'skip2': skip2}

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model with validation and early stopping"""
        self._initialize_weights(X.shape[1])

        n_samples = X.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        no_improve_counter = 0

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]

                # Forward pass
                predictions, cache = self.forward(batch_X, training=True)

                # Compute loss with L2 regularization
                mse_loss = np.mean((predictions - batch_y) ** 2)
                l2_loss = self.weight_decay * sum(np.sum(w ** 2) for w in self.weights.values())
                loss = mse_loss + l2_loss

                total_loss += loss
                num_batches += 1

                # Backward pass
                self._backward(batch_X, batch_y, predictions, cache)

            # Calculate average training loss
            avg_train_loss = total_loss / num_batches
            self.train_losses.append(avg_train_loss)

            # Validation loss
            if X_val is not None and y_val is not None:
                val_predictions, _ = self.forward(X_val, training=False)
                val_loss = np.mean((val_predictions - y_val) ** 2)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Learning rate decay
                if patience_counter >= self.lr_decay_patience:
                    self.learning_rate *= self.lr_decay_factor
                    patience_counter = 0
                    print(f"Learning rate decayed to: {self.learning_rate:.6f}")

            if epoch % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.learning_rate:.6f}")
                else:
                    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, LR: {self.learning_rate:.6f}")

    def _backward(self, X, y, predictions, cache):
        """Maximum overfitting backward pass"""
        m = X.shape[0]

        # Output layer gradient (W5)
        dz5 = (predictions - y) / m
        dW5 = np.dot(cache['a4'].T, dz5)
        db5 = np.sum(dz5, axis=0, keepdims=True)
        da4 = np.dot(dz5, self.weights['W5'].T)

        # Layer 4 with residual (a4 = leaky_relu(z4 + skip2))
        dz4_with_residual = da4 * self._leaky_relu_derivative(cache['z4'] + cache['skip2'])
        dW4 = np.dot(cache['a3'].T, dz4_with_residual)
        db4 = np.sum(dz4_with_residual, axis=0, keepdims=True)
        da3 = np.dot(dz4_with_residual, self.weights['W4'].T)

        # Residual gradient for Wr2 (64 -> 32)
        dWr2 = np.dot(cache['a1'].T, dz4_with_residual)
        da1_from_skip2 = np.dot(dz4_with_residual, self.weights['Wr2'].T)

        # Layer 3 with residual (a3 = leaky_relu(z3 + skip1))
        dz3_with_residual = da3 * self._leaky_relu_derivative(cache['z3'] + cache['skip1'])
        dW3 = np.dot(cache['a2'].T, dz3_with_residual)
        db3 = np.sum(dz3_with_residual, axis=0, keepdims=True)
        da2 = np.dot(dz3_with_residual, self.weights['W3'].T)

        # Residual gradient for Wr1 (64 -> 64)
        dWr1 = np.dot(cache['a1'].T, dz3_with_residual)
        da1_from_skip1 = np.dot(dz3_with_residual, self.weights['Wr1'].T)

        # Layer 2 gradients
        dz2 = da2 * self._leaky_relu_derivative(cache['z2'])
        dW2 = np.dot(cache['a1'].T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1_from_layer2 = np.dot(dz2, self.weights['W2'].T)

        # Combine gradients for a1 (from all sources)
        da1_total = da1_from_skip1 + da1_from_skip2 + da1_from_layer2

        # Layer 1 gradients
        dz1 = da1_total * self._leaky_relu_derivative(cache['z1'])
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights (no regularization for maximum overfitting)
        self.weights['W1'] -= self.learning_rate * dW1
        self.weights['W2'] -= self.learning_rate * dW2
        self.weights['W3'] -= self.learning_rate * dW3
        self.weights['W4'] -= self.learning_rate * dW4
        self.weights['W5'] -= self.learning_rate * dW5
        self.weights['Wr1'] -= self.learning_rate * dWr1
        self.weights['Wr2'] -= self.learning_rate * dWr2

        self.biases['b1'] -= self.learning_rate * db1
        self.biases['b2'] -= self.learning_rate * db2
        self.biases['b3'] -= self.learning_rate * db3
        self.biases['b4'] -= self.learning_rate * db4
        self.biases['b5'] -= self.learning_rate * db5

    def predict(self, X):
        """Make predictions"""
        predictions, _ = self.forward(X, training=False)
        return predictions

class CameraToJointRegressor:
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the regression model based on type"""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(32, 64, 128, 128, 64, 32),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        elif self.model_type == 'residual':
            self.model = ResidualMLPRegressor(learning_rate=0.001, epochs=3000, batch_size=8,
                                            dropout_rate=0.0, weight_decay=0.0,
                                            early_stopping_patience=500)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_data(self, data_path=None):
        """Load and combine all CSV files"""
        if data_path is None:
            # Find all unified_log CSV files
            csv_files = glob.glob('unified_log_*.csv')
        else:
            csv_files = [data_path]

        if not csv_files:
            raise ValueError("No CSV files found")

        print(f"📁 Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {file}")

        # Load and combine all data
        all_data = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                print(f"  ✓ {file}: {len(df)} rows")
                all_data.append(df)
            except Exception as e:
                print(f"  ❌ {file}: Failed to load ({e})")

        if not all_data:
            raise ValueError("No valid data files loaded")

        # Combine all dataframes
        self.raw_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total combined data: {len(self.raw_data)} rows")

        return self.raw_data

    def preprocess_data(self):
        """Preprocess the data for training"""
        print("🔧 Preprocessing data...")

        # Define input features (camera coordinates) and targets (joint positions)
        feature_cols = ['cam_left_x', 'cam_left_y', 'cam_right_x', 'cam_right_y']
        target_cols = ['follower_pos1', 'follower_pos2', 'follower_pos3', 'follower_pos4']

        # Remove rows with any missing values
        data_clean = self.raw_data.dropna(subset=feature_cols + target_cols)
        print(f"📊 After removing NaN values: {len(data_clean)} rows ({len(self.raw_data) - len(data_clean)} removed)")

        if len(data_clean) == 0:
            raise ValueError("No valid data remaining after preprocessing")

        # Extract features and targets
        X = data_clean[feature_cols].values
        y = data_clean[target_cols].values

        # Check for any remaining invalid values
        X_valid = np.isfinite(X).all(axis=1)
        y_valid = np.isfinite(y).all(axis=1)
        valid_mask = X_valid & y_valid

        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📊 Final dataset: {len(X)} samples with {X.shape[1]} features -> {y.shape[1]} targets")
        print(f"📈 Feature ranges:")
        for i, col in enumerate(feature_cols):
            print(f"  {col}: {X[:, i].min():.1f} - {X[:, i].max():.1f}")
        print(f"📈 Target ranges:")
        for i, col in enumerate(target_cols):
            print(f"  {col}: {y[:, i].min():.1f} - {y[:, i].max():.1f}")

        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        return X, y

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        print(f"📊 Train set: {len(self.X_train)} samples")
        print(f"📊 Test set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, normalize=True):
        """Train the regression model"""
        print(f"🎯 Training {self.model_type} regression model...")

        if normalize:
            # Normalize features and targets
            X_train_scaled = self.scaler_X.fit_transform(self.X_train)
            y_train_scaled = self.scaler_y.fit_transform(self.y_train)

            # For residual model, also scale validation set
            if self.model_type == 'residual':
                X_val_scaled = self.scaler_X.transform(self.X_test)
                y_val_scaled = self.scaler_y.transform(self.y_test)
        else:
            X_train_scaled = self.X_train
            y_train_scaled = self.y_train

            if self.model_type == 'residual':
                X_val_scaled = self.X_test
                y_val_scaled = self.y_test

        # Train the model
        if self.model_type == 'residual':
            # Custom residual model with validation
            self.model.fit(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        else:
            # Standard sklearn models
            self.model.fit(X_train_scaled, y_train_scaled)

        self.normalize = normalize

        print("✅ Training completed")

        # Print feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            print("📊 Feature importance:")
            for i, importance in enumerate(self.model.feature_importances_):
                print(f"  {self.feature_cols[i]}: {importance:.4f}")

    def predict(self, X):
        """Make predictions"""
        if self.normalize:
            X_scaled = self.scaler_X.transform(X)
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        else:
            y_pred = self.model.predict(X)

        return y_pred

    def evaluate(self):
        """Evaluate the model on test set"""
        print("📊 Evaluating model performance...")

        # Make predictions
        y_pred = self.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"📈 Overall Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")

        # Per-target metrics
        print(f"📈 Per-Joint Performance:")
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
        """Plot prediction results"""
        if not hasattr(self, 'test_metrics'):
            print("❌ No test metrics available. Run evaluate() first.")
            return

        y_test = self.test_metrics['y_test']
        y_pred = self.test_metrics['y_pred']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, target_col in enumerate(self.target_cols):
            ax = axes[i]

            # Scatter plot of actual vs predicted
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(y_test[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{target_col} (R² = {r2_score(y_test[:, i], y_pred[:, i]):.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"regression_results_{self.model_type}_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_filename}")

        plt.show()

    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_to_joint_model_{self.model_type}_{timestamp}.pkl"

        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model_type': self.model_type,
            'normalize': self.normalize,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols
        }

        joblib.dump(model_data, filename)
        print(f"💾 Model saved: {filename}")
        return filename

    def load_model(self, filename):
        """Load a trained model"""
        model_data = joblib.load(filename)

        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.model_type = model_data['model_type']
        self.normalize = model_data['normalize']
        self.feature_cols = model_data['feature_cols']
        self.target_cols = model_data['target_cols']

        print(f"📁 Model loaded: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Train camera to joint regression model")
    parser.add_argument("--data", help="Specific CSV file to use (default: all unified_log_*.csv)")
    parser.add_argument("--model", choices=['linear', 'rf', 'mlp', 'residual'], default='linear',
                       help="Model type: linear, rf (Random Forest), mlp (Neural Network), residual (Custom Residual MLP)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--no-normalize", action='store_true',
                       help="Disable feature normalization")
    parser.add_argument("--no-plot", action='store_true',
                       help="Skip plotting results")
    parser.add_argument("--output", help="Output model filename")

    args = parser.parse_args()

    print("🤖 Camera to Joint Regression Model Training")
    print("=" * 50)

    try:
        # Initialize regressor
        regressor = CameraToJointRegressor(model_type=args.model)

        # Load and preprocess data
        regressor.load_data(args.data)
        regressor.preprocess_data()
        regressor.split_data(test_size=args.test_size)

        # Train model
        regressor.train(normalize=not args.no_normalize)

        # Evaluate model
        regressor.evaluate()

        # Plot results
        if not args.no_plot:
            regressor.plot_results()

        # Save model
        regressor.save_model(args.output)

        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()