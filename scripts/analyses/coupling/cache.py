"""TE/Granger p-value cache to avoid recomputing expensive permutation tests.

Caches results keyed by (regime, seed, family_id, src_var, tgt_var, lag, bins).
Used when only aggregation parameters change (e.g. D1 weights, D3 edge modes)
but the underlying TE/Granger p-values are identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experiments" / ".te_cache"


def _cache_key(
    regime: str,
    seed: int,
    family_id: int,
    src_var: str,
    tgt_var: str,
    lag: int,
    bins: int,
) -> str:
    """Build a deterministic cache key string."""
    raw = f"{regime}:{seed}:{family_id}:{src_var}:{tgt_var}:lag{lag}:bins{bins}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    regime: str,
    seed: int,
    family_id: int,
    src_var: str,
    tgt_var: str,
    lag: int,
    bins: int,
) -> dict | None:
    """Return cached TE/Granger result or None if not cached."""
    key = _cache_key(regime, seed, family_id, src_var, tgt_var, lag, bins)
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def put_cached(
    regime: str,
    seed: int,
    family_id: int,
    src_var: str,
    tgt_var: str,
    lag: int,
    bins: int,
    result: dict,
) -> None:
    """Store TE/Granger result in cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(regime, seed, family_id, src_var, tgt_var, lag, bins)
    path = _CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(result, f)


def clear_cache() -> int:
    """Remove all cached entries. Returns count of files removed."""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for p in _CACHE_DIR.glob("*.json"):
        p.unlink()
        count += 1
    return count
