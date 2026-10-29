"""Portable fixed-recipe retraining of the earlier H4 champion.

No study modules, datasets or checkpoints are imported. Fit receives fitting
features/targets only; fixed epoch counts require no validation targets.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .models import build_model
except ImportError:
    from models import build_model


SEEDS = (19, 43, 101)
FIXED_EPOCHS = (35, 1, 15)


def recipe_metadata() -> dict[str, Any]:
    return {"interpretation": "fixed earlier-champion recipe retraining; original checkpoints unavailable; no claim of exact old-forecast replication",
            "ensemble": {"tree": 0.5, "residual_neural": 0.5},
            "ridge_base": {"alpha": 10, "fit_intercept": True,
                           "preprocessing": "mean imputer(add_indicator=True,keep_empty_features=True), then StandardScaler"},
            "neural_preprocessing": "separate identical imputer/scaler fitted only to fitting features; float32 inputs",
            "neural_architecture": "Linear(d,32), SiLU, Dropout(0.2), Linear(32,16), SiLU, Linear(16,1)",
            "neural_head_initialization": "final linear weights and bias zero",
            "neural_target": "in-sample residual (y-ridge.predict(X_fit))/100, float32",
            "neural_prediction": "ridge prediction + 100*mean(three fixed-seed neural residual predictions)",
            "seeds": list(SEEDS), "fixed_full_batch_steps": list(FIXED_EPOCHS),
            "validation_checkpoint_selection": False,
            "optimizer": {"name": "AdamW", "lr": 0.002, "weight_decay": 0.1,
                          "betas": [0.9, 0.999], "eps": 1e-8},
            "neural_loss": "unweighted mean squared residual error",
            "tree": {"loss": "squared_error", "max_iter": 150, "learning_rate": 0.05,
                     "max_leaf_nodes": 7, "min_samples_leaf": 30, "l2_regularization": 10,
                     "early_stopping": False, "random_state": 2026},
            "sample_weighting": "none"}


def _features(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not all(x.shape) or np.isinf(x).any():
        raise ValueError("Features must be a nonempty matrix with finite or missing values")
    return x


class PriorChampion:
    """Half shallow tree, half three-seed ridge-residual neural ensemble."""

    def __init__(self, device: str = "auto") -> None:
        self.device_request = device

    def fit(self, X: Any, y: Any) -> "PriorChampion":
        import torch
        from torch import nn

        x, target = _features(X), np.asarray(y, dtype=float)
        if target.shape != (len(x),) or not np.isfinite(target).all():
            raise ValueError("Fitting targets must be finite and aligned")
        self.n_features_in_ = x.shape[1]
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu") if self.device_request == "auto" else torch.device(self.device_request)
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.ridge_ = build_model("ridge_alpha_10")
        self.ridge_.fit(x, target)
        base_fit = self.ridge_.predict(x)
        self.neural_preprocessor_ = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler())])
        z = self.neural_preprocessor_.fit_transform(x).astype(np.float32)
        residual = ((target - base_fit) / 100.0).astype(np.float32)
        if not np.isfinite(z).all() or not np.isfinite(residual).all():
            raise ValueError("Nonfinite neural inputs or residuals")
        tensor_x = torch.as_tensor(z, dtype=torch.float32, device=self.device_)
        tensor_y = torch.as_tensor(residual, dtype=torch.float32, device=self.device_)
        self.networks_ = []
        self.fit_loss_ = []
        for seed, epochs in zip(SEEDS, FIXED_EPOCHS):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            network = nn.Sequential(nn.Linear(z.shape[1], 32), nn.SiLU(), nn.Dropout(0.2),
                                    nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1)).to(self.device_)
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.1)
            network.train()
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((network(tensor_x).squeeze(-1) - tensor_y) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite neural fitting loss")
                loss.backward()
                optimizer.step()
            network.eval()
            self.networks_.append(network)
            self.fit_loss_.append(float(loss.detach().cpu()))
        self.tree_ = build_model("hgb_leaves_7_minleaf_30")
        self.tree_.fit(x, target)
        self.runtime_ = {"device": str(self.device_), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "neural_input_columns_after_imputation": int(z.shape[1]),
                         "last_training_loss_by_seed": self.fit_loss_}
        return self

    def predict_components(self, X: Any) -> dict[str, np.ndarray]:
        import torch
        if not hasattr(self, "networks_"):
            raise ValueError("Fit the fixed recipe before predicting")
        x = _features(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting")
        z = self.neural_preprocessor_.transform(x).astype(np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(z, dtype=torch.float32, device=self.device_)
            residuals = np.stack([network(tensor).squeeze(-1).cpu().numpy() for network in self.networks_])
        ridge, tree = self.ridge_.predict(x), self.tree_.predict(x)
        neural = ridge + 100.0 * residuals.mean(axis=0)
        result = {"ridge": ridge, "tree": tree, "residual_neural": neural,
                  "prediction": 0.5 * tree + 0.5 * neural}
        result.update({f"neural_seed_{seed}": ridge + 100.0 * residuals[i] for i, seed in enumerate(SEEDS)})
        if any(not np.isfinite(values).all() for values in result.values()):
            raise ValueError("Fixed prior-champion recipe produced nonfinite forecasts")
        return result

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_components(X)["prediction"]


def fit_prior_champion(X: Any, y: Any, device: str = "auto") -> PriorChampion:
    """Fit only the supplied fitting rows; no validation/checkpoint interface."""
    return PriorChampion(device=device).fit(X, y)
