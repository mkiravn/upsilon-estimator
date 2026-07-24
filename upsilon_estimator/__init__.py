"""upsilon_estimator: Estimate upsilon_k (marker-causal relatedness ratio) from founder panels.

Core functions for loading zarr panels, drawing paired locus samples, and estimating
both crude (debiased r2 + Phi_k) and full (exact h_k components) upsilon_k values.
"""

from .estimators import estimate_upsilon_k, estimate_upsilon_k_crude, load_panel
from .utils import paired_draws, estimate_components

__all__ = ["estimate_upsilon_k", "estimate_upsilon_k_crude", "load_panel", "paired_draws", "estimate_components"]
