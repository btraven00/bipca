# omni-bipca

BiPCA (biwhitened PCA, KlugerLab) as an omnibenchmark **CNTFCT** module.

Sinkhorn biwhitening under a fitted quadratic mean-variance model, SVD, then
truncation at the Marchenko-Pastur bulk edge. The variance model is defined on
raw counts, which is why this sits on CNTFCT (`rawdata_h5ad`) and not on PCA
(`normalized_selected_h5` only). Cells come from `filtered_cellids`, genes from
the rownames of `normalized_selected_h5` — the same submatrix glmpca, newwave
and scGBM get.

    python pca.py --output_dir out --name be1 \
      --rawdata_h5ad be1.h5ad --filtered_cellids be1_cellids.txt.gz \
      --normalized_selected_h5 be1_normalized_selected.h5 --random_seed 42

Emits `{name}_pcas.tsv`, `{name}_loadings.tsv` and `{name}_bipca.json`
(diagnostics; not a declared stage output).

## k is derived

`k = mp_rank`, the number of singular values above the MP edge — the method's
whole contribution — so there is no `--n_components`. `--k_max` bounds the SVD
and the run aborts if `mp_rank` reaches it. A natural-k module and a fixed-k
module are not producing the same object; compare accordingly, and note the
metrics still do not report the k found.

## Gotchas

- BiPCA's Sinkhorn emits **NaN, not an exception**, on near-empty columns. The
  `--min_gene_cells` guard (default 10) drops them up front; the run aborts on
  any non-finite output and names the flag. 50 was needed at n=14000 cells.
- `--variance_estimator binomial` needs per-cell read counts BiPCA cannot get
  from this stage's inputs; it will fail. Left exposed rather than hidden.
- Torch is a hard dependency of the PyPI package, so the env carries it even
  though everything here runs on CPU.

`python tests/smoke.py` runs the whole path on a synthetic NB count matrix.
