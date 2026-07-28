"""upsilon_estimator: Estimate upsilon_k (marker-causal relatedness ratio) from founder panels.

Core functions for loading zarr panels, drawing paired locus samples, and estimating
upsilon_k using the robust model: (r²_MQ + Φₖ) / (r²_MM + Φₖ).

The robust model drops the cross term (B_k r·λ_a·λ_b) because:
  - It contributes <0.5% signal but has kurtosis in the thousands
  - Including it inflates estimates by 13-39% with no gain
  - It is sign-cancelling and rarely informative

Primary API:
  - estimate_upsilon_k(): Main estimator (debiased r² + shared IBD, no cross term)
  - estimate_upsilon_k_weighted(): Speed alpha-model weighting (frequency-dependent variance)
  - estimate_upsilon_k_crude(): Deprecated alias for estimate_upsilon_k()
  - estimate_upsilon_k_full(): Deprecated; includes noisy cross term

See upsilon_k_decomposition.md for theoretical justification.
"""

from .estimators import (
    estimate_upsilon_k,
    estimate_upsilon_k_weighted,
    estimate_upsilon_k_crude,
    estimate_upsilon_k_full,
    load_panel,
)
from .utils import paired_draws, estimate_components

__all__ = [
    "estimate_upsilon_k",
    "estimate_upsilon_k_weighted",
    "estimate_upsilon_k_crude",
    "estimate_upsilon_k_full",
    "load_panel",
    "paired_draws",
    "estimate_components",
]
