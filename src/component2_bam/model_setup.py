"""
model_setup.py
--------------
BAM architecture, unchanged from the original notebook (WavLM-base-plus +
Boundary Enhancement + Boundary Framewise Attention + utterance/segment
heads), plus config-driven build helpers for the model, optimizer,
scheduler, loss criteria, and feature extractor.

Testability note: BAM's constructor accepts an optional `backbone`
argument. When None (the normal/Colab path), it loads the real SSL model
via `WavLMModel.from_pretrained(ssl_model_name)`. Tests inject a tiny
randomly-initialized WavLMModel(WavLMConfig(...)) instead, so the full
architecture (BE/BFA/heads) can be exercised end to end offline, with no
network access and no pretrained-weight download. This does not change
the architecture used in Colab in any way - it only decouples "which
weights populate self.wavlm" from "does the rest of the graph work".
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# Architecture (unchanged from the BAM_v2 notebook)
# ============================================================================

class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, return_weights: bool = False):
        # x: (B, L, D)
        w = torch.softmax(self.attn(x), dim=1)
        pooled = (w * x).sum(dim=1)
        if return_weights:
            return pooled, w.squeeze(-1)   # (B, D), (B, L)
        return pooled


class FramewiseAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn: bool = False):
        attn_out, attn_weights = self.attn(
            x, x, x, attn_mask=attn_mask,
            need_weights=return_attn, average_attn_weights=False,
        )
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        if return_attn:
            return x, attn_weights   # (B, n_heads, L, L) or None
        return x


class BoundaryEnhancementModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
        )
        self.boundary_clf = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(dim, 1),
        )

    def forward(self, x):   # x: (B, L, D)
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        diff = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        fused = torch.cat([x_conv, diff], dim=-1)
        boundary_logits = self.boundary_clf(fused).squeeze(-1)
        return boundary_logits


class BoundaryFramewiseAttentionModule(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.layers = nn.ModuleList([
            FramewiseAttentionBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])

    def _build_mask(self, boundary_probs):
        B, L = boundary_probs.shape
        is_bdy = (boundary_probs.detach() > 0.5).float()
        seg_ids = torch.cumsum(is_bdy, dim=1)
        blocked = (seg_ids.unsqueeze(2) != seg_ids.unsqueeze(1)).float()
        mask = blocked * -1e9
        mask = mask.unsqueeze(1).expand(B, self.n_heads, L, L)
        return mask.reshape(B * self.n_heads, L, L)

    def forward(self, x, boundary_probs=None, return_attn: bool = False):
        attn_mask = self._build_mask(boundary_probs) if boundary_probs is not None else None
        attn_maps = [] if return_attn else None
        for layer in self.layers:
            if return_attn:
                x, w = layer(x, attn_mask=attn_mask, return_attn=True)
                attn_maps.append(w)
            else:
                x = layer(x, attn_mask=attn_mask)
        if return_attn:
            return x, attn_maps
        return x


def build_boundary_features(boundary_probs: torch.Tensor,
                             peak_threshold: float = 0.5,
                             window: int = 5) -> torch.Tensor:
    """Three per-frame features derived from the BE module's boundary
    probabilities, for feeding directly to the segment head.

    Motivated by measurement, not intuition. XAI on the eval split found
    that the BE module localizes true segment edges accurately (median
    boundary probability 0.762 at true offsets vs a 0.0000 baseline both
    inside spoof spans and far from them; 79.4% of TOO_LONG segments have
    offset probability > 0.5) while the segment head over-extends past
    those edges anyway. The reason is structural: boundary_probs reach the
    rest of the network through exactly one path -

        boundary_probs -> (> 0.5) -> cumsum -> segment IDs -> BFA attn mask

    - so the frame classifier never sees the boundary signal directly, and
    the hard > 0.5 step throws away its magnitude (0.76 and 0.51 become
    identical). These three features restore that information:

      [0] prob        the continuous boundary probability at frame t.
                      Recovers the magnitude the binary mask discards.

      [1] since_peak  frames elapsed since the last frame whose probability
                      exceeded peak_threshold, normalised. Answers "have I
                      already crossed a boundary?" - the TOO_LONG case
                      (19.7% of reachable segments).

      [2] local_max   max probability within +/-window frames. Answers "is
                      there a boundary anywhere near me?" - the FRAGMENTED
                      case (10.8%), where the head splits a run at a point
                      the BE module scores 0.0000.

    All three are computed from boundary_probs with no new parameters, so
    the only trainable change is the widened seg_head input.

    boundary_probs: (B, L). Returns (B, L, 3).
    """
    B, L = boundary_probs.shape
    device = boundary_probs.device

    prob = boundary_probs

    # frames since the last peak. Detached: these features are meant to
    # route existing boundary information into the frame decision, not to
    # let the seg_head loss reshape boundary detection itself - that would
    # make this a two-variable change and confound the comparison.
    is_peak = (boundary_probs.detach() > peak_threshold)
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    # index of the most recent peak at or before each frame (-1 if none yet)
    peak_idx = torch.where(is_peak, idx, torch.full_like(idx, -1))
    last_peak = torch.cummax(peak_idx, dim=1).values
    since_peak = (idx - last_peak).float()
    # frames with no preceding peak get the full elapsed distance, which is
    # already what (idx - (-1)) gives, so no special case is needed.
    since_peak = (since_peak / L).clamp(0.0, 1.0)

    # local maximum probability in a +/-window neighbourhood
    pad = window
    local_max = torch.nn.functional.max_pool1d(
        torch.nn.functional.pad(prob.unsqueeze(1), (pad, pad), mode="replicate"),
        kernel_size=2 * window + 1, stride=1,
    ).squeeze(1)

    return torch.stack([prob, since_peak, local_max], dim=-1)


class BAM(nn.Module):
    def __init__(self, ssl_model_name: str, hidden_dim: int = 256,
                 n_heads: int = 4, n_bfa_layers: int = 2,
                 layer_weighting: bool = False,
                 dropout: float = 0.1, freeze_cnn: bool = True,
                 backbone=None, boundary_gated_seg_head: bool = False,
                 boundary_peak_threshold: float = 0.5,
                 boundary_window: int = 5):
        """backbone: optional pre-built SSL module with the same interface
        as transformers.WavLMModel (forward(x).last_hidden_state, plus
        .config.hidden_size and .feature_extractor.parameters()). When
        None, loads WavLMModel.from_pretrained(ssl_model_name) - the normal
        path. Tests pass a tiny randomly-initialized WavLMModel instead,
        to exercise this whole class without network access.
        """
        super().__init__()
        if backbone is not None:
            self.wavlm = backbone
        else:
            from transformers import WavLMModel
            self.wavlm = WavLMModel.from_pretrained(ssl_model_name)

        if freeze_cnn:
            for p in self.wavlm.feature_extractor.parameters():
                p.requires_grad = False

        ssl_dim = self.wavlm.config.hidden_size

        # H6: learned softmax-weighted sum over all WavLM hidden states.
        # Default False keeps the original last-layer behaviour byte for byte.
        self.layer_weighting = bool(layer_weighting)
        if self.layer_weighting:
            n_hidden = self.wavlm.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(n_hidden))
        self.proj = nn.Sequential(
            nn.Linear(ssl_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.be_module = BoundaryEnhancementModule(hidden_dim)
        self.bfa_module = BoundaryFramewiseAttentionModule(hidden_dim, n_heads, n_bfa_layers, dropout)
        self.attentive_pool = AttentivePool(hidden_dim)
        self.utt_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

        # H1: when boundary_gated_seg_head is on, the segment head also takes
        # the three boundary-derived features (see build_boundary_features).
        # This is the ONLY architectural difference between the baseline and
        # the H1 experiment - utt_head, BE, BFA and pooling are untouched, so
        # any change in results is attributable to this one thing.
        self.boundary_gated_seg_head = boundary_gated_seg_head
        self.boundary_peak_threshold = boundary_peak_threshold
        self.boundary_window = boundary_window
        self.n_boundary_features = 3 if boundary_gated_seg_head else 0
        self.seg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + self.n_boundary_features, 2),
        )

    def forward(self, input_values, return_xai: bool = False):
        if self.layer_weighting:
            hs = self.wavlm(input_values, output_hidden_states=True).hidden_states
            hs = torch.stack(hs, dim=0)                    # (L+1, B, T, D)
            w = torch.softmax(self.layer_weights, dim=0)   # (L+1,)
            hidden = (w.view(-1, 1, 1, 1) * hs).sum(0)     # (B, T, D)
        else:
            hidden = self.wavlm(input_values).last_hidden_state
        x = self.proj(hidden)
        boundary_logits = self.be_module(x)
        boundary_probs = torch.sigmoid(boundary_logits)

        if return_xai:
            x, bfa_attn_maps = self.bfa_module(x, boundary_probs, return_attn=True)
            pooled, pool_weights = self.attentive_pool(x, return_weights=True)
            utt_logits = self.utt_head(pooled)
            seg_in, bfeat = self._seg_head_input(x, boundary_probs)
            seg_logits = self.seg_head(seg_in)
            xai = {
                "bfa_attn_maps": bfa_attn_maps,     # list[len n_bfa_layers] of (B, n_heads, L, L)
                "pool_weights": pool_weights,        # (B, L)
                "bfa_hidden": x.detach(),            # (B, L, D)
                "boundary_features": None if bfeat is None else bfeat.detach(),
            }
            return utt_logits, seg_logits, boundary_logits, xai

        x = self.bfa_module(x, boundary_probs)
        utt_logits = self.utt_head(self.attentive_pool(x))
        seg_in, _ = self._seg_head_input(x, boundary_probs)
        seg_logits = self.seg_head(seg_in)
        return utt_logits, seg_logits, boundary_logits

    def _seg_head_input(self, x, boundary_probs):
        """Returns (seg_head input, boundary features or None). Keeping this
        in one place means the plain and return_xai paths cannot drift apart.
        """
        if not self.boundary_gated_seg_head:
            return x, None
        bfeat = build_boundary_features(
            boundary_probs, peak_threshold=self.boundary_peak_threshold,
            window=self.boundary_window)
        return torch.cat([x, bfeat], dim=-1), bfeat

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ============================================================================
# Config-driven build helpers
# ============================================================================

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config: dict, backbone_override=None) -> BAM:
    m = config["model"]
    device = get_device()
    model = BAM(
        ssl_model_name=m["ssl_model"], hidden_dim=m["hidden_dim"],
        n_heads=m["n_heads"], n_bfa_layers=m["n_bfa_layers"],
        layer_weighting=m.get("layer_weighting", False),
        freeze_cnn=m["freeze_cnn"], backbone=backbone_override,
        boundary_gated_seg_head=m.get("boundary_gated_seg_head", False),
        boundary_peak_threshold=m.get("boundary_peak_threshold", 0.5),
        boundary_window=m.get("boundary_window", 5),
    )
    return model.to(device)


def build_optimizer(model: BAM, config: dict) -> torch.optim.Optimizer:
    o = config["optim"]
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=o["lr"], weight_decay=o["weight_decay"],
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    epochs = config["train"]["epochs"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def build_criteria(config: dict) -> dict:
    return {
        "utt": nn.CrossEntropyLoss(),
        "seg": nn.CrossEntropyLoss(),
        "bdy": nn.BCEWithLogitsLoss(),
    }


def build_feature_extractor(config: dict, override=None):
    """override: injectable stand-in with the same call signature as
    transformers.Wav2Vec2FeatureExtractor (waveforms list -> object with
    .input_values), for offline testing.
    """
    if override is not None:
        return override
    from transformers import Wav2Vec2FeatureExtractor
    return Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["ssl_model"])


def preprocess_batch(waveforms, feature_extractor, sample_rate: int, device: str):
    inputs = feature_extractor(
        [w.numpy() for w in waveforms],
        sampling_rate=sample_rate, return_tensors="pt", padding=True,
    )
    return inputs.input_values.to(device)


def load_model_for_eval(config: dict, checkpoint_path, backbone_override=None):
    """Build a fresh BAM per `config` and load weights from
    `checkpoint_path`. Used both by evaluate.py (this experiment's own
    best.pth) and by the baseline-import verification notebook (an
    arbitrary external checkpoint path, e.g. the frozen bam_baseline_legacy
    weights).
    """
    import utils   # local import to avoid a hard circular-import at module load time

    model = build_model(config, backbone_override=backbone_override)
    utils.load_checkpoint(checkpoint_path, model, map_location=get_device())
    model.eval()
    return model
