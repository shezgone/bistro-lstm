"""
LSTM Trainer — Walk-Forward CV + Optuna Tuning
===============================================
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

from lstm_model import AttentionLSTMForecaster
from lstm_core import LSTMConfig


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: AttentionLSTMForecaster,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """단일 에폭 학습. Gaussian NLL loss 사용."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = model(X_batch)
        mu = out["mu"]
        log_sigma = out["log_sigma"]

        # Gaussian NLL: 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        loss = nn.GaussianNLLLoss()(mu, y_batch, sigma ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: AttentionLSTMForecaster,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """검증 세트 평가. RMSE, MAE, NLL 반환."""
    model.eval()
    all_mu = []
    all_y = []
    total_nll = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        out = model(X_batch)
        mu = out["mu"]
        log_sigma = out["log_sigma"]

        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        nll = nn.GaussianNLLLoss()(mu, y_batch, sigma ** 2)
        total_nll += nll.item()
        n_batches += 1

        all_mu.append(mu.cpu().numpy())
        all_y.append(y_batch.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    rmse = np.sqrt(np.mean((all_mu - all_y) ** 2))
    mae = np.mean(np.abs(all_mu - all_y))

    return {
        "rmse": rmse,
        "mae": mae,
        "nll": total_nll / max(n_batches, 1),
    }


def train_model(
    model: AttentionLSTMForecaster,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 20,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """
    단일 fold 학습.

    Returns
    -------
    dict with best_model_state, train_history, val_history, best_val_rmse
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # DataLoader 생성
    train_ds = TensorDataset(
        torch.FloatTensor(train_X),
        torch.FloatTensor(train_y),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(val_X),
        torch.FloatTensor(val_y),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    early_stop = EarlyStopping(patience=patience)
    train_history = []
    val_history = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        train_history.append(train_loss)
        val_history.append(val_metrics)

        improved = early_stop.step(val_metrics["rmse"], model)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or improved or early_stop.should_stop):
            best_marker = " ★ best" if early_stop.counter == 0 else ""
            print(
                f"  Epoch {epoch+1:3d}/{max_epochs} | "
                f"Train NLL: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Best: {early_stop.best_loss:.4f} "
                f"(patience {early_stop.counter}/{patience}){best_marker}"
            )

        if improved:
            break

    # Restore best model
    if early_stop.best_model_state is not None:
        model.load_state_dict(early_stop.best_model_state)

    return {
        "best_model_state": early_stop.best_model_state or model.state_dict(),
        "train_history": train_history,
        "val_history": val_history,
        "best_val_rmse": early_stop.best_loss,
    }


def walk_forward_cv(
    splits: List[Dict],
    config: LSTMConfig,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 20,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """
    Walk-forward cross-validation.

    Parameters
    ----------
    splits : prepare_walk_forward_splits()의 출력
    config : LSTMConfig
    기타 : 학습 하이퍼파라미터

    Returns
    -------
    dict with fold_results, mean_val_rmse, best_fold
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []

    for split in splits:
        fold = split["fold"]
        val_year = split["val_year"]
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold}: Train → {split['train_end']}, Val: {val_year}")
            print(f"  Train: {split['train_X'].shape}, Val: {split['val_X'].shape}")

        set_seed(42 + fold)

        model = AttentionLSTMForecaster.from_config(config)
        result = train_model(
            model=model,
            train_X=split["train_X"],
            train_y=split["train_y"],
            val_X=split["val_X"],
            val_y=split["val_y"],
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            verbose=verbose,
        )

        result["fold"] = fold
        result["val_year"] = val_year
        fold_results.append(result)

        if verbose:
            print(f"  Best Val RMSE: {result['best_val_rmse']:.4f}")

    val_rmses = [r["best_val_rmse"] for r in fold_results]
    mean_rmse = np.mean(val_rmses)
    best_fold = int(np.argmin(val_rmses))

    if verbose:
        print(f"\n{'='*50}")
        print(f"Mean Val RMSE: {mean_rmse:.4f}")
        print(f"Per-fold RMSE: {[f'{r:.4f}' for r in val_rmses]}")

    return {
        "fold_results": fold_results,
        "mean_val_rmse": mean_rmse,
        "best_fold": best_fold,
        "val_rmses": val_rmses,
    }


def optuna_objective(
    trial,
    splits: List[Dict],
    base_variates: List[str],
    device: torch.device = None,
) -> float:
    """
    Optuna objective function.
    Walk-forward CV의 평균 val RMSE를 최소화.
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 4)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    # hidden_dim must be divisible by n_heads (for temporal decoder)
    if hidden_dim % n_heads != 0:
        raise ValueError("hidden_dim must be divisible by n_heads")

    config = LSTMConfig(
        variates=base_variates,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
    )

    cv_result = walk_forward_cv(
        splits=splits,
        config=config,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=200,
        patience=15,
        device=device,
        verbose=False,
    )

    return cv_result["mean_val_rmse"]


def run_optuna_tuning(
    splits: List[Dict],
    base_variates: List[str],
    n_trials: int = 50,
    device: torch.device = None,
) -> Dict:
    """
    Optuna 하이퍼파라미터 튜닝.

    Returns
    -------
    dict with best_params, best_value, study
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial):
        try:
            return optuna_objective(trial, splits, base_variates, device)
        except ValueError:
            return float("inf")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  Value (Mean Val RMSE): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
