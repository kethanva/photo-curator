"""
Aesthetic scoring using CLIP zero-shot semantic comparison.

Works from pre-computed CLIP embeddings stored in the database —
no additional model inference required at scoring time.

Two approaches available:
  1. CLIP zero-shot (default) — uses stored embeddings, instant, no extra deps
  2. LAION Improved Aesthetic Predictor (optional) — downloads a small MLP
     trained on the AVA dataset; more accurate but requires internet on first run.
     Enable via config: aesthetic.use_laion_predictor: true

The CLIP approach works by computing cosine similarity between the image
embedding and averaged embeddings of positive/negative aesthetic prompts.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Prompt bank — curated for consistent aesthetic direction
# ---------------------------------------------------------------------------

_POSITIVE_PROMPTS = [
    "a stunning professional photograph",
    "beautiful award-winning photography",
    "a perfectly composed, sharply focused photo",
    "a vibrant, colourful, high-resolution photograph",
    "an artistic portrait with beautiful lighting",
]

_NEGATIVE_PROMPTS = [
    "a blurry out-of-focus snapshot",
    "a grainy noisy low-quality photo",
    "a dark underexposed photograph",
    "an overexposed washed-out photo",
    "a badly composed amateur snapshot",
]

# Cached averaged text feature vectors
_pos_vec: Optional[np.ndarray] = None
_neg_vec: Optional[np.ndarray] = None


def _build_prompt_vectors() -> None:
    global _pos_vec, _neg_vec
    if _pos_vec is not None:
        return

    import torch
    import clip
    from src.embeddings import load_model

    model, _, device = load_model()

    def _encode(prompts: List[str]) -> np.ndarray:
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        avg = feats.mean(dim=0)
        avg = avg / avg.norm()
        return avg.cpu().numpy().astype(np.float32)

    _pos_vec = _encode(_POSITIVE_PROMPTS)
    _neg_vec = _encode(_NEGATIVE_PROMPTS)


# ---------------------------------------------------------------------------
# LAION Improved Aesthetic Predictor (optional)
# ---------------------------------------------------------------------------

_laion_model = None


def _load_laion_predictor():
    """
    Download and cache the LAION aesthetic MLP head.
    Requires: pip install huggingface_hub torch
    Model: ~6 KB linear layer trained on AVA aesthetic ratings.
    """
    global _laion_model
    if _laion_model is not None:
        return _laion_model

    import torch
    import torch.nn as nn
    from pathlib import Path

    # Simple linear model matching the improved-aesthetic-predictor weights
    class AestheticPredictor(nn.Module):
        def __init__(self, input_size: int = 512):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.layers(x)

    cache_path = Path(__file__).parent.parent / "models" / "aesthetic_predictor.pth"
    cache_path.parent.mkdir(exist_ok=True)

    if not cache_path.exists():
        print("  Downloading aesthetic predictor weights…")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            downloaded = hf_hub_download(
                repo_id="christophschuhmann/improved-aesthetic-predictor",
                filename="sac+logos+ava1-l14-linearMSE.pth",
                local_dir=str(cache_path.parent),
            )
            shutil.move(downloaded, cache_path)
        except Exception as e:
            print(f"  Warning: could not download aesthetic model ({e}). Falling back to CLIP.")
            return None

    try:
        m = AestheticPredictor(512)
        state = torch.load(cache_path, map_location="cpu")
        m.load_state_dict(state, strict=False)
        m.eval()
        _laion_model = m
        return m
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_from_embedding(
    clip_emb: np.ndarray,
    use_laion: bool = False,
) -> float:
    """
    Return aesthetic score [0, 1] for a pre-computed CLIP embedding.

    Args:
        clip_emb: normalised 512-dim float32 CLIP embedding
        use_laion: if True, attempt to use the LAION predictor MLP

    Returns:
        Aesthetic quality score [0, 1].
    """
    if clip_emb is None or len(clip_emb) != 512:
        return 0.5

    if use_laion:
        model = _load_laion_predictor()
        if model is not None:
            try:
                import torch
                x = torch.tensor(clip_emb, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    raw = model(x).item()  # Scale ≈ 1–10
                return float(np.clip((raw - 1.0) / 9.0, 0.0, 1.0))
            except Exception:
                pass  # Fall through to CLIP zero-shot

    # CLIP zero-shot approach
    try:
        _build_prompt_vectors()
        emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        pos = float(np.dot(emb, _pos_vec))
        neg = float(np.dot(emb, _neg_vec))
        # Map [-1, 1] diff to [0, 1]
        raw = (pos - neg + 2.0) / 4.0
        return float(np.clip(raw, 0.0, 1.0))
    except Exception:
        return 0.5


def batch_score(
    clip_embs: np.ndarray,
    use_laion: bool = False,
) -> np.ndarray:
    """Score a matrix of CLIP embeddings (N, 512). Returns float32 array (N,)."""
    return np.array(
        [score_from_embedding(clip_embs[i], use_laion) for i in range(len(clip_embs))],
        dtype=np.float32,
    )
