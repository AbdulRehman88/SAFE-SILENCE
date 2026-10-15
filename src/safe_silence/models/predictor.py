"""Task I compact causal prediction and calibrated environmental uncertainty."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_silence.data.preprocess import sha256_file
from safe_silence.reproducibility import seed_everything


class PredictorError(RuntimeError):
    """Raised when Task I cannot preserve its leakage or calibration contract."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def verify_task_h_inputs(input_root: Path) -> tuple[dict[str, Any], int]:
    summary_path = input_root / "labeling_summary.json"
    if not summary_path.exists():
        raise PredictorError(f"Missing Task H summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("task") != "H" or summary.get("labeling_version") != "1.1":
        raise PredictorError("Task I requires the locked Task H labeling contract v1.1")
    data_root = input_root.parent.parent
    verified = 0
    for artifact in summary.get("artifacts", []):
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not relative or not expected:
            raise PredictorError("Task H artifact lacks path or SHA-256")
        path = data_root / str(relative)
        if not path.exists() or sha256_file(path) != str(expected):
            raise PredictorError(f"Task H artifact changed or is missing: {path}")
        verified += 1
    if verified == 0:
        raise PredictorError("Task H summary contains no verifiable artifacts")
    return summary, verified


def assign_model_roles(frame: pd.DataFrame, tuning_fraction: float) -> pd.DataFrame:
    """Split validation windows chronologically without consulting measurements."""
    required = {"window_key", "timestamp", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise PredictorError(f"Role assignment missing columns: {sorted(missing)}")
    if not 0.0 < tuning_fraction < 1.0:
        raise PredictorError("model.validation_roles.tuning_fraction must be in (0, 1)")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["model_role"] = result["split"]
    validation = result[result["split"] == "validation"]
    ordered_windows = (
        validation.groupby("window_key", sort=False)["timestamp"].min().sort_values().index.tolist()
    )
    if len(ordered_windows) < 2:
        raise PredictorError("At least two validation windows are required for disjoint roles")
    tuning_count = int(math.floor(len(ordered_windows) * tuning_fraction))
    tuning_count = min(max(tuning_count, 1), len(ordered_windows) - 1)
    tuning_windows = set(ordered_windows[:tuning_count])
    calibration_windows = set(ordered_windows[tuning_count:])
    if tuning_windows.intersection(calibration_windows):
        raise PredictorError("Tuning and calibration windows overlap")
    result.loc[result["window_key"].isin(tuning_windows), "model_role"] = "tuning"
    result.loc[result["window_key"].isin(calibration_windows), "model_role"] = "calibration"
    return result


def fit_observed_scaler(
    frame: pd.DataFrame, channels: list[str], fit_site: str, fit_split: str
) -> dict[str, Any]:
    """Fit channel moments only on observed values from the locked site/split."""
    fit = frame[(frame["site"] == fit_site) & (frame["split"] == fit_split)]
    if fit.empty:
        raise PredictorError("No rows satisfy the locked scaler fit site/split")
    output: dict[str, Any] = {
        "fit_site": fit_site,
        "fit_split": fit_split,
        "observed_only": True,
        "channels": {},
    }
    for channel in channels:
        observed = fit[f"observed_{channel}"].fillna(0).astype(bool)
        values = pd.to_numeric(fit.loc[observed, channel], errors="coerce").dropna()
        if len(values) < 2:
            raise PredictorError(f"Too few observed scaler values for {channel}")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise PredictorError(f"Degenerate training scale for {channel}")
        output["channels"][channel] = {
            "mean": mean,
            "scale": scale,
            "observations": int(len(values)),
        }
    return output


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int, float]:
    """Finite-sample split-conformal quantile using the conservative higher rule."""
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        raise PredictorError("Too few finite calibration scores")
    if not 0.0 < alpha < 1.0:
        raise PredictorError("Conformal alpha must be in (0, 1)")
    rank = int(math.ceil((len(finite) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(finite))
    quantile_level = rank / len(finite)
    value = float(np.quantile(finite, quantile_level, method="higher"))
    return value, rank, quantile_level


def worst_window_conformal(
    scores: np.ndarray,
    window_keys: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
    minimum_windows: int,
) -> dict[str, Any]:
    """Use the most conservative finite-sample quantile across calibration windows."""
    scores = np.asarray(scores, dtype=float)
    windows = np.asarray(window_keys)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(scores)
    per_window: dict[str, Any] = {}
    for window in pd.unique(windows[valid]):
        selected = valid & (windows == window)
        value, rank, level = conformal_quantile(scores[selected], alpha)
        per_window[str(window)] = {
            "multiplier": value,
            "observations": int(selected.sum()),
            "finite_sample_rank": rank,
            "quantile_level": level,
        }
    if len(per_window) < minimum_windows:
        raise PredictorError(
            f"Only {len(per_window)} calibration windows; require {minimum_windows}"
        )
    worst_key = max(per_window, key=lambda key: per_window[key]["multiplier"])
    pooled_value, pooled_rank, pooled_level = conformal_quantile(scores[valid], alpha)
    return {
        "multiplier": float(per_window[worst_key]["multiplier"]),
        "worst_window": worst_key,
        "calibration_windows": len(per_window),
        "calibration_observations": int(valid.sum()),
        "pooled_multiplier_diagnostic": pooled_value,
        "pooled_finite_sample_rank": pooled_rank,
        "pooled_quantile_level": pooled_level,
        "per_window": per_window,
    }


@dataclass(frozen=True)
class PreparedWindow:
    features: np.ndarray
    target_normalized: np.ndarray
    target_original: np.ndarray
    target_mask: np.ndarray
    persistence_original: np.ndarray
    metadata: pd.DataFrame


class WindowSequenceDataset:
    """Lazy within-window sequences; samples can never cross a window boundary."""

    def __init__(
        self,
        frame: pd.DataFrame,
        channels: list[str],
        scaler: dict[str, Any],
        input_steps: int,
        age_cap_steps: int,
        roles: set[str] | None = None,
        require_observed_target: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise PredictorError(
                "PyTorch is unavailable; install the verified CUDA build before Task I"
            ) from exc
        self.torch = torch
        self.channels = channels
        self.input_steps = int(input_steps)
        selected = frame if roles is None else frame[frame["model_role"].isin(roles)]
        self.windows: list[PreparedWindow] = []
        self.samples: list[tuple[int, int]] = []
        metadata_columns = [
            "protocol",
            "site",
            "split",
            "model_role",
            "window_key",
            "timestamp",
        ]
        metadata_columns.extend(
            column
            for column in ("label_critical_event", "valid_critical_event")
            if column in selected.columns
        )
        for _, group in selected.groupby("window_key", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) <= self.input_steps:
                continue
            normalized_values = np.zeros((len(group), len(channels)), dtype=np.float32)
            target_mask = np.zeros_like(normalized_values, dtype=bool)
            original = np.zeros_like(normalized_values, dtype=np.float32)
            feature_parts: list[np.ndarray] = []
            observed_parts: list[np.ndarray] = []
            imputed_parts: list[np.ndarray] = []
            age_parts: list[np.ndarray] = []
            for channel_index, channel in enumerate(channels):
                values = pd.to_numeric(group[channel], errors="coerce").to_numpy(dtype=float)
                observed = group[f"observed_{channel}"].fillna(0).to_numpy(
                    dtype=bool
                ) & np.isfinite(values)
                imputed = group[f"imputed_{channel}"].fillna(0).to_numpy(dtype=np.float32)
                age = pd.to_numeric(group[f"age_steps_{channel}"], errors="coerce").fillna(
                    age_cap_steps + 1
                )
                age = np.minimum(age.to_numpy(dtype=float), age_cap_steps + 1)
                mean = float(scaler["channels"][channel]["mean"])
                scale = float(scaler["channels"][channel]["scale"])
                normalized = (values - mean) / scale
                target_normalized = normalized.copy()
                target_normalized[~np.isfinite(target_normalized)] = 0.0
                normalized_values[:, channel_index] = target_normalized.astype(np.float32)
                original[:, channel_index] = values.astype(np.float32)
                target_mask[:, channel_index] = observed
                causal_feature = pd.Series(normalized).ffill().fillna(0.0).to_numpy()
                feature_parts.append(causal_feature.astype(np.float32)[:, None])
                observed_parts.append(observed.astype(np.float32)[:, None])
                imputed_parts.append(imputed[:, None])
                age_parts.append((age / (age_cap_steps + 1)).astype(np.float32)[:, None])
            features = np.concatenate(
                [
                    np.concatenate(feature_parts, axis=1),
                    np.concatenate(observed_parts, axis=1),
                    np.concatenate(imputed_parts, axis=1),
                    np.concatenate(age_parts, axis=1),
                ],
                axis=1,
            )
            persistence = np.full_like(original, np.nan, dtype=np.float32)
            for channel_index in range(len(channels)):
                last = np.nan
                for row_index in range(len(group)):
                    if np.isfinite(original[row_index, channel_index]):
                        last = original[row_index, channel_index]
                    persistence[row_index, channel_index] = last
            prepared = PreparedWindow(
                features=features,
                target_normalized=normalized_values,
                target_original=original,
                target_mask=target_mask,
                persistence_original=persistence,
                metadata=group[metadata_columns].copy(),
            )
            window_index = len(self.windows)
            self.windows.append(prepared)
            for target_index in range(self.input_steps, len(group)):
                if require_observed_target and not target_mask[target_index].any():
                    continue
                self.samples.append((window_index, target_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, int]:
        window_index, target_index = self.samples[index]
        window = self.windows[window_index]
        start = target_index - self.input_steps
        x = window.features[start:target_index].T
        y = window.target_normalized[target_index]
        mask = window.target_mask[target_index]
        return (
            self.torch.from_numpy(x),
            self.torch.from_numpy(y),
            self.torch.from_numpy(mask),
            index,
        )

    def sample_table(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for sample_index, (window_index, target_index) in enumerate(self.samples):
            window = self.windows[window_index]
            row = window.metadata.iloc[target_index].to_dict()
            row["sample_index"] = sample_index
            for channel_index, channel in enumerate(self.channels):
                row[f"actual_{channel}"] = float(
                    window.target_original[target_index, channel_index]
                )
                row[f"target_observed_{channel}"] = int(
                    window.target_mask[target_index, channel_index]
                )
                row[f"persistence_{channel}"] = float(
                    window.persistence_original[target_index - 1, channel_index]
                )
            records.append(row)
        return pd.DataFrame(records)


def _build_tcn(model_config: dict[str, Any], input_features: int, outputs: int) -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise PredictorError("PyTorch is unavailable") from exc

    class CausalBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
            super().__init__()
            kernel = int(model_config["kernel_size"])
            self.padding = (kernel - 1) * dilation
            self.conv = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel,
                dilation=dilation,
                padding=self.padding,
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(float(model_config["dropout"]))
            self.residual = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv1d(input_channels, output_channels, kernel_size=1)
            )

        def forward(self, value: Any) -> Any:
            transformed = self.conv(value)
            if self.padding:
                transformed = transformed[:, :, : -self.padding]
            transformed = self.dropout(self.activation(transformed))
            return self.activation(transformed + self.residual(value))

    class EnvironmentalTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            current = input_features
            for index, hidden in enumerate(model_config["hidden_channels"]):
                layers.append(CausalBlock(current, int(hidden), dilation=2**index))
                current = int(hidden)
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Linear(current, outputs * 2)
            nn.init.zeros_(self.head.weight[:outputs])
            nn.init.zeros_(self.head.bias[:outputs])

        def forward(self, value: Any) -> tuple[Any, Any]:
            encoded = self.encoder(value)[:, :, -1]
            output = self.head(encoded)
            residual, log_variance = torch.chunk(output, 2, dim=1)
            persistence = value[:, :outputs, -1]
            return persistence + residual, log_variance.clamp(min=-8.0, max=5.0)

    return EnvironmentalTCN()


def _masked_gaussian_nll(mean: Any, log_variance: Any, target: Any, mask: Any) -> Any:
    import torch

    weights = mask.to(dtype=mean.dtype)
    loss = 0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _loader(dataset: WindowSequenceDataset, config: dict[str, Any], shuffle: bool) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=False,
    )


def _validation_loss(model: Any, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    total = 0.0
    weight = 0
    with torch.no_grad():
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            mean, log_variance = model(x)
            count = int(mask.sum().item())
            total += float(_masked_gaussian_nll(mean, log_variance, y, mask).item()) * count
            weight += count
    if weight == 0:
        raise PredictorError("Validation role has no observed prediction targets")
    return total / weight


def train_member(
    train_dataset: WindowSequenceDataset,
    tuning_dataset: WindowSequenceDataset,
    model_config: dict[str, Any],
    seed: int,
    device: Any,
) -> tuple[Any, list[dict[str, Any]], int]:
    import torch

    seed_everything(seed, deterministic_torch=True)
    model = _build_tcn(
        model_config,
        input_features=len(model_config["channels"]) * len(model_config["features"]),
        outputs=len(model_config["channels"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > int(model_config["maximum_parameters"]):
        raise PredictorError(
            f"Predictor has {parameters} parameters, above locked maximum "
            f"{model_config['maximum_parameters']}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["training"]["learning_rate"]),
        weight_decay=float(model_config["training"]["weight_decay"]),
    )
    train_loader = _loader(train_dataset, model_config, shuffle=True)
    tuning_loader = _loader(tuning_dataset, model_config, shuffle=False)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    patience = 0
    history: list[dict[str, Any]] = []
    maximum_epochs = int(model_config["training"]["maximum_epochs"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        running = 0.0
        observations = 0
        for x, y, mask, _ in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, log_variance = model(x)
            loss = _masked_gaussian_nll(mean, log_variance, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(model_config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            count = int(mask.sum().item())
            running += float(loss.item()) * count
            observations += count
        tuning_loss = _validation_loss(model, tuning_loader, device)
        training_loss = running / max(observations, 1)
        history.append(
            {
                "epoch": epoch,
                "training_nll": training_loss,
                "tuning_nll": tuning_loss,
            }
        )
        if tuning_loss < best_loss - 1e-6:
            best_loss = tuning_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(model_config["training"]["early_stopping_patience"]):
                break
    if best_state is None:
        raise PredictorError("Training did not produce a finite tuning checkpoint")
    model.load_state_dict(best_state)
    return model, history, parameters


def predict_ensemble(
    models: list[Any], dataset: WindowSequenceDataset, model_config: dict[str, Any], device: Any
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    loader = _loader(dataset, model_config, shuffle=False)
    member_means: list[np.ndarray] = []
    member_variances: list[np.ndarray] = []
    for model in models:
        model.eval()
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                mean, log_variance = model(x.to(device))
                means.append(mean.cpu().numpy())
                variances.append(torch.exp(log_variance).cpu().numpy())
        member_means.append(np.concatenate(means, axis=0))
        member_variances.append(np.concatenate(variances, axis=0))
    stacked_means = np.stack(member_means, axis=0)
    stacked_variances = np.stack(member_variances, axis=0)
    ensemble_mean = stacked_means.mean(axis=0)
    total_variance = (stacked_variances + stacked_means**2).mean(axis=0) - ensemble_mean**2
    return ensemble_mean, np.maximum(total_variance, 1e-8)


def _calibrate(
    dataset: WindowSequenceDataset,
    mean: np.ndarray,
    variance: np.ndarray,
    channels: list[str],
    alpha: float,
    minimum_windows: int,
) -> dict[str, Any]:
    table = dataset.sample_table()
    result: dict[str, Any] = {
        "method": "worst_window_normalized_split_conformal",
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "channels": {},
    }
    for channel_index, channel in enumerate(channels):
        mask = table[f"target_observed_{channel}"].to_numpy(dtype=bool)
        scaler_target = np.array(
            [
                dataset.windows[window_index].target_normalized[target_index, channel_index]
                for window_index, target_index in dataset.samples
            ]
        )
        score = np.abs(scaler_target - mean[:, channel_index]) / np.sqrt(variance[:, channel_index])
        result["channels"][channel] = worst_window_conformal(
            score,
            table["window_key"].to_numpy(),
            mask,
            alpha,
            minimum_windows,
        )
    return result


def _prediction_table(
    dataset: WindowSequenceDataset,
    mean_normalized: np.ndarray,
    variance_normalized: np.ndarray,
    scaler: dict[str, Any],
    calibration: dict[str, Any],
) -> pd.DataFrame:
    table = dataset.sample_table()
    for channel_index, channel in enumerate(dataset.channels):
        center = float(scaler["channels"][channel]["mean"])
        scale = float(scaler["channels"][channel]["scale"])
        mean = mean_normalized[:, channel_index] * scale + center
        standard_deviation = np.sqrt(variance_normalized[:, channel_index]) * scale
        multiplier = float(calibration["channels"][channel]["multiplier"])
        table[f"prediction_{channel}"] = mean.astype(np.float32)
        table[f"uncertainty_std_{channel}"] = standard_deviation.astype(np.float32)
        table[f"lower_{channel}"] = (mean - multiplier * standard_deviation).astype(np.float32)
        table[f"upper_{channel}"] = (mean + multiplier * standard_deviation).astype(np.float32)
    return table


def _metric_rows(table: pd.DataFrame, channels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_columns = ["protocol", "site", "model_role"]
    for keys, group in table.groupby(group_columns, sort=False):
        conditions = {"overall": np.ones(len(group), dtype=bool)}
        if "label_critical_event" in group:
            conditions["critical_event"] = group["label_critical_event"].fillna(0).to_numpy(
                dtype=bool
            ) & group["valid_critical_event"].fillna(0).to_numpy(dtype=bool)
        for condition_name, condition in conditions.items():
            for channel in channels:
                observed = group[f"target_observed_{channel}"].to_numpy(dtype=bool)
                mask = observed & condition
                actual = group[f"actual_{channel}"].to_numpy(dtype=float)
                prediction = group[f"prediction_{channel}"].to_numpy(dtype=float)
                lower = group[f"lower_{channel}"].to_numpy(dtype=float)
                upper = group[f"upper_{channel}"].to_numpy(dtype=float)
                finite = mask & np.isfinite(actual) & np.isfinite(prediction)
                if finite.any():
                    error = prediction[finite] - actual[finite]
                    coverage = (actual[finite] >= lower[finite]) & (actual[finite] <= upper[finite])
                    rows.append(
                        {
                            "protocol": keys[0],
                            "site": keys[1],
                            "model_role": keys[2],
                            "condition": condition_name,
                            "model": "tcn_deep_ensemble",
                            "channel": channel,
                            "observations": int(finite.sum()),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "mae": float(np.mean(np.abs(error))),
                            "interval_coverage": float(np.mean(coverage)),
                            "mean_interval_width": float(np.mean(upper[finite] - lower[finite])),
                        }
                    )
                persistence = group[f"persistence_{channel}"].to_numpy(dtype=float)
                persistence_finite = mask & np.isfinite(actual) & np.isfinite(persistence)
                if persistence_finite.any():
                    error = persistence[persistence_finite] - actual[persistence_finite]
                    rows.append(
                        {
                            "protocol": keys[0],
                            "site": keys[1],
                            "model_role": keys[2],
                            "condition": condition_name,
                            "model": "persistence",
                            "channel": channel,
                            "observations": int(persistence_finite.sum()),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "mae": float(np.mean(np.abs(error))),
                            "interval_coverage": np.nan,
                            "mean_interval_width": np.nan,
                        }
                    )
    return rows


def train_predictor(
    input_root: Path,
    artifact_root: Path,
    result_root: Path,
    model_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Train, tune, calibrate, and evaluate Task I without touching test information."""
    try:
        import torch
    except ImportError as exc:
        raise PredictorError("PyTorch is unavailable; install the verified CUDA build") from exc
    _, verified = verify_task_h_inputs(input_root)
    channels = [str(value) for value in model_config["channels"]]
    if int(model_config["forecast_steps"]) != 1:
        raise PredictorError("Task I currently locks one-step causal forecasts")
    requested_device = str(model_config["training"]["device"])
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PredictorError("CUDA was requested but is unavailable")
    base_seed = int(project_config["seed"])
    all_metrics: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    protocol_summaries: dict[str, Any] = {}

    for protocol_index, (protocol, protocol_config) in enumerate(
        preprocessing_config["protocols"].items()
    ):
        sites = [protocol_config["development_site"], *protocol_config["external_sites"]]
        frames: dict[str, pd.DataFrame] = {}
        for site in sites:
            path = input_root / protocol / f"{site}.parquet"
            if not path.exists():
                raise PredictorError(f"Missing Task H dataset: {path}")
            frame = pd.read_parquet(path)
            frame = (
                assign_model_roles(
                    frame, float(model_config["validation_roles"]["tuning_fraction"])
                )
                if site == protocol_config["development_site"]
                else frame.assign(model_role=frame["split"])
            )
            frames[site] = frame

        development = frames[protocol_config["development_site"]]
        scaler = fit_observed_scaler(
            development,
            channels,
            str(model_config["scaling"]["fit_site"]),
            str(model_config["scaling"]["fit_split"]),
        )
        protocol_artifact_root = artifact_root / protocol
        scaler_path = protocol_artifact_root / "scaler.json"
        _atomic_json(scaler_path, scaler)

        dataset_arguments = {
            "channels": channels,
            "scaler": scaler,
            "input_steps": int(model_config["input_steps_by_protocol"][protocol]),
            "age_cap_steps": int(preprocessing_config["age_cap_steps"]),
        }
        training_dataset = WindowSequenceDataset(
            development, **dataset_arguments, roles={"train"}, require_observed_target=True
        )
        tuning_dataset = WindowSequenceDataset(
            development, **dataset_arguments, roles={"tuning"}, require_observed_target=True
        )
        calibration_dataset = WindowSequenceDataset(
            development,
            **dataset_arguments,
            roles={"calibration"},
            require_observed_target=False,
        )
        if min(len(training_dataset), len(tuning_dataset), len(calibration_dataset)) == 0:
            raise PredictorError(f"Empty Task I role for {protocol}")

        models: list[Any] = []
        parameter_count = 0
        member_paths: list[str] = []
        for member in range(int(model_config["uncertainty"]["members"])):
            member_seed = base_seed + protocol_index * 1000 + member
            model, history, parameter_count = train_member(
                training_dataset, tuning_dataset, model_config, member_seed, device
            )
            models.append(model)
            for record in history:
                all_history.append(
                    {"protocol": protocol, "member": member, "seed": member_seed, **record}
                )
            member_path = protocol_artifact_root / f"ensemble_member_{member:02d}.pt"
            member_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = member_path.with_suffix(".pt.part")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "channels": channels,
                    "seed": member_seed,
                },
                temporary,
            )
            os.replace(temporary, member_path)
            member_paths.append(str(member_path))
            artifacts.append(
                {
                    "protocol": protocol,
                    "member": member,
                    "path": str(member_path),
                    "sha256": sha256_file(member_path),
                }
            )

        calibration_mean, calibration_variance = predict_ensemble(
            models, calibration_dataset, model_config, device
        )
        calibration = _calibrate(
            calibration_dataset,
            calibration_mean,
            calibration_variance,
            channels,
            float(model_config["uncertainty"]["miscoverage_alpha"]),
            int(model_config["uncertainty"]["minimum_calibration_windows"]),
        )
        calibration_path = protocol_artifact_root / "conformal_calibration.json"
        _atomic_json(calibration_path, calibration)
        artifacts.extend(
            [
                {
                    "protocol": protocol,
                    "path": str(scaler_path),
                    "sha256": sha256_file(scaler_path),
                },
                {
                    "protocol": protocol,
                    "path": str(calibration_path),
                    "sha256": sha256_file(calibration_path),
                },
            ]
        )

        role_counts = (
            development.groupby("model_role")["window_key"].nunique().astype(int).to_dict()
        )
        for site, frame in frames.items():
            prediction_dataset = WindowSequenceDataset(
                frame, **dataset_arguments, roles=None, require_observed_target=False
            )
            mean, variance = predict_ensemble(models, prediction_dataset, model_config, device)
            table = _prediction_table(prediction_dataset, mean, variance, scaler, calibration)
            output_path = result_root / "predictions" / protocol / f"{site}.parquet"
            _atomic_parquet(output_path, table)
            all_metrics.extend(_metric_rows(table, channels))
            artifacts.append(
                {
                    "protocol": protocol,
                    "site": site,
                    "path": str(output_path),
                    "rows": int(len(table)),
                    "sha256": sha256_file(output_path),
                }
            )
        protocol_summaries[protocol] = {
            "grid_minutes": int(protocol_config["grid_minutes"]),
            "history_hours": float(model_config["history_hours"]),
            "input_steps": int(model_config["input_steps_by_protocol"][protocol]),
            "development_site": protocol_config["development_site"],
            "external_sites": protocol_config["external_sites"],
            "role_window_counts": role_counts,
            "training_samples": len(training_dataset),
            "tuning_samples": len(tuning_dataset),
            "calibration_samples": len(calibration_dataset),
            "ensemble_members": len(models),
            "parameters_per_member": parameter_count,
            "member_paths": member_paths,
        }
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(all_metrics)
    history = pd.DataFrame(all_history)
    metrics_path = result_root / "predictive_metrics.csv"
    history_path = result_root / "training_history.csv"
    _atomic_csv(metrics_path, metrics)
    _atomic_csv(history_path, history)
    artifacts.extend(
        [
            {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            {"path": str(history_path), "sha256": sha256_file(history_path)},
        ]
    )
    config_hash = hashlib.sha256(
        json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "task": "I",
        "predictor_version": "1.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "task_h_artifacts_reverified": verified,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scaler_fit_site": model_config["scaling"]["fit_site"],
        "scaler_fit_split": model_config["scaling"]["fit_split"],
        "validation_used_for": "chronologically_disjoint_tuning_and_calibration",
        "prediction_parameterization": model_config["prediction_parameterization"],
        "uncertainty_calibration": model_config["uncertainty"]["calibration"],
        "calibration_used_for_early_stopping": False,
        "test_or_external_data_used_for_training_tuning_or_calibration": False,
        "temporal_protocols_may_be_pooled": False,
        "protocols": protocol_summaries,
        "artifacts": artifacts,
    }
    _atomic_json(result_root / "task_i_summary.json", summary)
    return summary
