"""PyTorch `nn.Module` for the supervised autoencoder + MLP.

This is the only module in the package that imports torch. Torch is treated
as an *optional* dependency:

- Module import is lazy: `from src.models.supervised_autoencoder_mlp import ...`
  does NOT pull in torch. The torch import happens the first time
  `build_network`, `train_one`, or `predict_proba` is called.
- Callers that don't want a hard torch dependency can use the helpers in
  `signal.py` (feature engineering, target binarization, purged K-fold,
  median blend, action rule) directly.

If torch is unavailable, the functions here raise `RuntimeError` with a
clear install hint.

The architecture matches the spec's canonical configuration:

    Input x in R^{p}
      |
      +-- Gaussian noise sigma -> x_tilde
                                      |
                                      v
                Encoder: Linear -> BN -> SiLU -> Dropout  (per spec)
                                      |
                                      v                            == h (bottleneck)
                 +-----------+--------------+--------------+
                 |                          |              |
            Decoder D_psi             Aux head           [x ; h] -> MLP -> y_hat
            Linear -> ... -> Linear   Linear -> sigmoid    (deeper MLP)
            == x_hat                  == y_aux

Joint loss `L = alpha_rec * MSE(x_hat, x) + alpha_aux * BCE(y_aux, y)
                + alpha_main * BCE(y_hat, y)`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.models.supervised_autoencoder_mlp.types import SAEConfig, TrainConfig

__all__ = [
    "build_network",
    "predict_proba",
    "train_one",
]


_TORCH_INSTALL_HINT = (
    "PyTorch is required for SAE+MLP training/inference but is not installed. "
    "Install with `pip install torch` (CPU build is sufficient for tests)."
)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError(_TORCH_INSTALL_HINT) from exc
    return torch


def _require_nn() -> tuple[Any, Any]:
    torch = _require_torch()
    return torch, torch.nn


def _make_dense_block(
    nn_mod: Any,
    in_dim: int,
    out_dim: int,
    dropout: float,
    *,
    use_dropout: bool = True,
) -> Any:
    """Linear -> BatchNorm -> SiLU -> optional Dropout."""

    layers: list[Any] = [
        nn_mod.Linear(in_dim, out_dim),
        nn_mod.BatchNorm1d(out_dim),
        nn_mod.SiLU(),
    ]
    if use_dropout and dropout > 0.0:
        layers.append(nn_mod.Dropout(p=dropout))
    return nn_mod.Sequential(*layers)


def build_network(config: SAEConfig) -> Any:
    """Construct the SAE + MLP `nn.Module`. Requires torch.

    Returns an `nn.Module` instance whose `forward(x_clean)` (eval) /
    `forward(x_clean, training_noise=True)` produces
    ``(y_hat, y_aux, x_hat, h)``.
    """

    if config.n_features < 1:
        raise ValueError(
            "SAEConfig.n_features must be set to a positive integer before "
            f"building the network; got {config.n_features}"
        )
    torch, nn = _require_nn()

    class SAEMLPNet(nn.Module):  # type: ignore[name-defined,misc]
        """Concrete network. Defined inside `build_network` to gate the torch import."""

        def __init__(self, cfg: SAEConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.noise_sigma = cfg.noise_sigma

            enc_layers: list[Any] = []
            prev = cfg.n_features
            for width in cfg.encoder_hidden:
                enc_layers.append(_make_dense_block(nn, prev, width, cfg.dropout_encoder))
                prev = width
            # Bottleneck: no dropout on h itself (preserves representation).
            enc_layers.append(_make_dense_block(nn, prev, cfg.bottleneck_dim, 0.0, use_dropout=False))
            self.encoder = nn.Sequential(*enc_layers)

            dec_layers: list[Any] = []
            prev = cfg.bottleneck_dim
            for width in cfg.decoder_hidden:
                dec_layers.append(_make_dense_block(nn, prev, width, cfg.dropout_encoder))
                prev = width
            dec_layers.append(nn.Linear(prev, cfg.n_features))
            self.decoder = nn.Sequential(*dec_layers)

            self.aux_head = nn.Linear(cfg.bottleneck_dim, cfg.n_targets)

            mlp_layers: list[Any] = []
            prev = cfg.n_features + cfg.bottleneck_dim
            for width in cfg.mlp_hidden:
                mlp_layers.append(_make_dense_block(nn, prev, width, cfg.dropout_mlp))
                prev = width
            mlp_layers.append(nn.Linear(prev, cfg.n_targets))
            self.main_head = nn.Sequential(*mlp_layers)

        def forward(
            self, x: Any, *, training_noise: bool = False
        ) -> tuple[Any, Any, Any, Any]:
            if training_noise and self.noise_sigma > 0:
                x_in = x + torch.randn_like(x) * self.noise_sigma
            else:
                x_in = x
            h = self.encoder(x_in)
            x_hat = self.decoder(h)
            y_aux_logit = self.aux_head(h)
            y_hat_logit = self.main_head(torch.cat([x, h], dim=1))
            return y_hat_logit, y_aux_logit, x_hat, h

    return SAEMLPNet(config)


def _to_tensor(arr: np.ndarray, torch_mod: Any, device: Any | None = None) -> Any:
    t = torch_mod.from_numpy(arr.astype(np.float32))
    if device is not None:
        t = t.to(device)
    return t


def train_one(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray | None,
    config: SAEConfig,
    train_cfg: TrainConfig,
    seed: int,
    device: str = "cpu",
) -> tuple[Any, float, float, list[float]]:
    """Train one SAE+MLP and return ``(state_dict, train_loss, val_loss, val_auc)``.

    A single (fold, seed) training run. The function:

    1. Seeds torch with `seed` for reproducibility.
    2. Builds the network from `config`.
    3. Runs `train_cfg.epochs` of Adam + cosine annealing.
    4. Reports the final-epoch *training* loss, the held-out *validation*
       loss, and per-target ROC-AUC on the validation set.

    Sample weights are applied to *both* the classification heads' BCE losses
    (not the reconstruction loss — the spec is explicit on this).
    """

    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("x_train and x_val must be 2-D")
    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError("y_train and y_val must be 2-D")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(
            f"feature dim mismatch: x_train {x_train.shape[1]} vs x_val {x_val.shape[1]}"
        )

    torch, nn = _require_nn()
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    network = build_network(config).to(dev)

    x_tr_t = _to_tensor(x_train, torch, dev)
    y_tr_t = _to_tensor(y_train, torch, dev)
    x_va_t = _to_tensor(x_val, torch, dev)
    y_va_t = _to_tensor(y_val, torch, dev)
    if sample_weights is None:
        w_tr_t = torch.ones(x_train.shape[0], device=dev)
    else:
        w_tr_t = _to_tensor(sample_weights, torch, dev)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(train_cfg.epochs, 1),
        eta_min=train_cfg.lr_min,
    )
    recon_loss_fn = nn.MSELoss(reduction="mean")
    # We compute BCE manually so sample weights enter cleanly.
    bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    n_train = x_train.shape[0]
    batch_size = min(train_cfg.batch_size, n_train)
    last_train_loss = float("nan")
    for _epoch in range(train_cfg.epochs):
        network.train()
        perm = torch.randperm(n_train, device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            # BatchNorm1d in training mode needs >= 2 samples for the variance
            # estimate. Skip any trailing micro-batch that would crash it.
            if idx.numel() < 2:
                continue
            xb = x_tr_t[idx]
            yb = y_tr_t[idx]
            wb = w_tr_t[idx]
            optimizer.zero_grad()
            y_hat_logit, y_aux_logit, x_hat, _ = network(xb, training_noise=True)
            l_rec = recon_loss_fn(x_hat, xb)
            l_aux_per = bce_logits(y_aux_logit, yb).mean(dim=1)
            l_main_per = bce_logits(y_hat_logit, yb).mean(dim=1)
            w_norm = wb / (wb.mean() + 1e-12)
            l_aux = (l_aux_per * w_norm).mean()
            l_main = (l_main_per * w_norm).mean()
            loss = (
                train_cfg.alpha_recon * l_rec
                + train_cfg.alpha_aux * l_aux
                + train_cfg.alpha_main * l_main
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
        scheduler.step()
        last_train_loss = epoch_loss / max(n_batches, 1)

    # Validation pass.
    network.eval()
    with torch.no_grad():
        y_hat_logit, y_aux_logit, x_hat, _ = network(x_va_t, training_noise=False)
        l_rec = recon_loss_fn(x_hat, x_va_t)
        l_aux = bce_logits(y_aux_logit, y_va_t).mean()
        l_main = bce_logits(y_hat_logit, y_va_t).mean()
        val_loss_t = (
            train_cfg.alpha_recon * l_rec
            + train_cfg.alpha_aux * l_aux
            + train_cfg.alpha_main * l_main
        )
        val_loss = float(val_loss_t.detach().cpu().item())
        # Per-target AUCs on the main head.
        probs = torch.sigmoid(y_hat_logit).detach().cpu().numpy()

    # Import here to avoid a circular import at module load.
    from src.models.supervised_autoencoder_mlp.signal import roc_auc

    val_auc = [roc_auc(probs[:, k], y_val[:, k]) for k in range(y_val.shape[1])]

    # Move state dict to CPU before returning so callers can move it freely.
    state_dict = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    return state_dict, float(last_train_loss), val_loss, val_auc


def predict_proba(
    *,
    state_dict: Any,
    x: np.ndarray,
    config: SAEConfig,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run a trained network in `eval` mode and return ``(probs, x_hat)``.

    `probs` is the main-head sigmoid output, shape (N, K). `x_hat` is the
    decoded reconstruction, shape (N, p). The reconstruction is returned so
    downstream consumers can compute reconstruction-error anomaly scores.
    """

    torch = _require_torch()
    network = build_network(config)
    network.load_state_dict(state_dict)
    network.eval()
    dev = torch.device(device)
    network = network.to(dev)
    x_t = _to_tensor(x, torch, dev)
    with torch.no_grad():
        y_hat_logit, _, x_hat, _ = network(x_t, training_noise=False)
        probs = torch.sigmoid(y_hat_logit).detach().cpu().numpy()
        x_hat_np = x_hat.detach().cpu().numpy()
    return cast(np.ndarray, probs), cast(np.ndarray, x_hat_np)


