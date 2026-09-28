# omni-bipca

BiPCA (biwhitened PCA, KlugerLab) as an omnibenchmark **RDIMR** (raw dimensionality reduction) module.

Sinkhorn biwhitening under a fitted quadratic mean-variance model, SVD, then
truncation at the Marchenko-Pastur bulk edge. The variance model is defined on
raw counts, which is why this sits on RDIMR (`rawdata_h5ad`) and not on PCA
(`normalized_selected_h5` only). Cells come from `filtered_cellids`, genes from
`filtered_featureids`: every gene FILT kept, no HVG selection, because BiPCA
needs the noise-dominated genes to place the MP edge.

    python pca.py --output_dir out --name be1 \
      --rawdata_h5ad be1.h5ad --filtered_cellids be1_cellids.txt.gz \
      --filtered_featureids be1_featureids.txt.gz --random_seed 42

Emits `{name}_embedding.tsv`, `{name}_loadings.tsv` and `{name}_bipca.json`
(diagnostics; not a declared stage output).

## k is derived

`k = mp_rank`, the number of singular values above the MP edge — the method's
whole contribution — so there is no `--n_components`. `--k_max` bounds the SVD
and the run aborts if `mp_rank` reaches it. A natural-k module and a fixed-k
module are not producing the same object; compare accordingly, and note the
metrics still do not report the k found.

## Measured on be1

RDIMR wiring (every FILT gene), omni, 4 threads, `--random_seed 42`, defaults:
1715 cells x 18896 genes; `--min_gene_cells 10` dropped 2346, leaving 16550.
Sinkhorn **converged** after 20 iterations, sigma 0.935, q 0.845,
**`mp_rank=32`**. 2m04s wall, 1.8 GB peak RSS.

On the old CNTFCT wiring (FEAT's 2000 genes) the same fixture gave
`mp_rank=25` and lost 810 of the 2000 genes to `--min_gene_cells`.

Worth contrasting with `pc-rmt-spca`, which biwhitens the same fixture and
never converged on it at any damping. Different variance model (quadratic fit
on counts vs. none) and different input (counts vs. log-normalized).

## Gotchas

- BiPCA's Sinkhorn emits **NaN, not an exception**, on near-empty columns. The
  `--min_gene_cells` guard (default 10) drops them up front; the run aborts on
  any non-finite output and names the flag. 50 was needed at n=14000 cells.
- Only BiPCA's `quadratic` variance estimator is used. `binomial` models each
  entry as Binomial(n, p) with a known per-entry trial count (`read_counts`,
  e.g. methylation coverage); UMI counts have none, and `BiPCA()` raises.
- Non-integer counts are refused: the variance model is only defined on counts.
- Torch is a hard dependency of the PyPI package, so the env carries it even
  though everything here runs on CPU.

`pixi run test` runs the whole path on a synthetic NB count matrix.
`pixi run export-env` regenerates `envs/bipca.yml`, which the plan copies
verbatim.
