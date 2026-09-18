"""Spot-check r² values by direct computation from genotypes."""
import numpy as np
from upsilon_estimator.estimators import load_panel
from upsilon_estimator._theory import r2_debiased
from upsilon_estimator._panel import _pairwise_r_from_rows, fetch_sites

PANEL_PATH = "/home/mianniravn/claude-822397757/-project2-jnovembre-mianniravn-glitch-0526/a892de0d-7b26-411d-b8bb-9a87234dd901/scratchpad/founders_n5000_gbr_converted.npz"

panel_data = load_panel(PANEL_PATH)
panel = panel_data["panel"]
af = panel_data["af"]
maf = np.minimum(af, 1.0 - af)
n_hap = panel_data["n_hap"]

print(f"Panel: n_sites={len(af)}, n_hap={n_hap}\n")

# Test sc3: markers MAF>1%, causals all variable sites
marker_idx = np.nonzero(maf > 0.01)[0]
causal_idx = np.nonzero(maf > 0)[0]

print(f"sc3_gcta spot-check:")
print(f"  markers (MAF>0.01): {len(marker_idx)}")
print(f"  causals (MAF>0): {len(causal_idx)}\n")

# Sample random pairs
rng = np.random.default_rng(42)
n_samples = 500  # Larger sample

# Marker-marker pairs
print("Computing marker-marker r²...")
mm_anchors = rng.choice(marker_idx, n_samples)
mm_partners = rng.choice(marker_idx, n_samples)
bad = mm_anchors == mm_partners
while bad.any():
    mm_partners[bad] = rng.choice(marker_idx, bad.sum())
    bad = mm_anchors == mm_partners

mm_geno1 = fetch_sites(panel, mm_anchors)
mm_geno2 = fetch_sites(panel, mm_partners)
mm_r = _pairwise_r_from_rows(mm_geno1, mm_geno2)
mm_r2_raw = mm_r ** 2
mm_r2_debiased = np.maximum(r2_debiased(mm_r2_raw, n_hap), 0)

print(f"  raw r² mean={np.mean(mm_r2_raw):.3e}, median={np.median(mm_r2_raw):.3e}, std={np.std(mm_r2_raw):.3e}")
print(f"  debiased r² mean={np.nanmean(mm_r2_debiased):.3e}, median={np.nanmedian(mm_r2_debiased):.3e}")

# Marker-causal pairs
print("\nComputing marker-causal r²...")
mq_anchors = rng.choice(marker_idx, n_samples)
mq_partners = rng.choice(causal_idx, n_samples)
bad = mq_anchors == mq_partners
while bad.any():
    mq_partners[bad] = rng.choice(causal_idx, bad.sum())
    bad = mq_anchors == mq_partners

mq_geno1 = fetch_sites(panel, mq_anchors)
mq_geno2 = fetch_sites(panel, mq_partners)
mq_r = _pairwise_r_from_rows(mq_geno1, mq_geno2)
mq_r2_raw = mq_r ** 2
mq_r2_debiased = np.maximum(r2_debiased(mq_r2_raw, n_hap), 0)

print(f"  raw r² mean={np.mean(mq_r2_raw):.3e}, median={np.median(mq_r2_raw):.3e}, std={np.std(mq_r2_raw):.3e}")
print(f"  debiased r² mean={np.nanmean(mq_r2_debiased):.3e}, median={np.nanmedian(mq_r2_debiased):.3e}")

print(f"\nRatio (debiased mean):")
print(f"  r²_MQ / r²_MM = {np.nanmean(mq_r2_debiased) / np.nanmean(mm_r2_debiased):.4f}")
print(f"\nRatio (raw mean):")
print(f"  r²_MQ / r²_MM = {np.mean(mq_r2_raw) / np.mean(mm_r2_raw):.4f}")
