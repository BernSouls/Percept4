"""perception/noise_floor.py -- the ONE robust noise-floor statistic.

[U30 R-D3] Every energy gate in the front end must answer the question
"is this reading distinguishable from this channel's own background, this
frame?" with the SAME statistic, measured on this frame, instead of a
global constant.

    med   = median(x)              (the channel's own central level)
    MAD   = median(|x - med|)      (raw median absolute deviation)
    floor = med + k * MAD          (a reading is evidence only above this)

`k` is recorded by the caller in parameters (k=3 for the endstop
energy gate; k_emit for the per-channel emission gates) — it is never a
free constant inside a gate.

`mad` is the RAW median absolute deviation, deliberately not the
1.4826*sigma-consistent normalisation: the statistic is a floor for
gating, not an estimator of sigma, and multiplying by a constant that
assumes Gaussianity would pretend the frame's noise is Gaussian when it
is measured, not assumed.

The function is total: empty / all-NaN / degenerate inputs return
(0.0, 0.0, 0.0) so a caller that has not inspected the return can still
clamp harmlessly — but callers SHOULD short-circuit on empty input.
Non-finite values are excluded from the statistics; if any finite value
remains, it is used.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def measured_floor(x: np.ndarray, k: float) -> Tuple[float, float, float]:
    """Return (med, mad, floor) for the finite values in `x`.

    floor = med + k * mad.  All three are 0.0 for empty / all-NaN /
    degenerate input.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, mad, med + float(k) * mad
