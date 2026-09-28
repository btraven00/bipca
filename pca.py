#!/usr/bin/env python3
"""RDIMR module: BiPCA (biwhitened PCA, Stanley et al. / KlugerLab).

BiPCA fits a quadratic mean-variance relationship (Poisson/NB-like), Sinkhorn-
biwhitens the matrix so the noise sits on the canonical Marchenko-Pastur scale,
SVDs it, and truncates at the MP bulk edge. The variance model is defined on
*raw counts*, so this lands on RDIMR -- the only stage that gets
`rawdata_h5ad` -- and not on PCA, whose single input is already normalized.

Cells come from --filtered_cellids, genes from the rownames of
--normalized_selected_h5: same subsetting the R cntfct module does, so BiPCA
and glmpca/newwave/scGBM see the same submatrix of counts.

Outputs
-------
{output_dir}/{name}_embedding.tsv    cell_id  PC1..PCk   (U * shrunk S)
{output_dir}/{name}_loadings.tsv     gene_id  PC1..PCk   (V, unit norm)
{output_dir}/{name}_bipca.json       mp_rank, q, KS, shapes  (diagnostics, not a stage output)

k is derived, not requested
---------------------------
k = mp_rank, the count of singular values above the MP bulk edge. That is the
method's contribution, so there is no --n_components target; --k_max is a
search ceiling and the run FAILS if k saturates it (a capped rank is the cap,
not the answer). Same convention as pc-rmt-spca.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package
from common import cli  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="BiPCA biwhitened PCA module")
    cli.add_base_args(p)             # --output_dir, --name
    cli.add_stage_args(p, "RDIMR")   # --rawdata_h5ad, --filtered_cellids, --normalized_selected_h5
    p.add_argument("--variance_estimator", choices=["quadratic", "binomial"],
                   default="quadratic",
                   help="quadratic = Poisson/NB-like mean-variance fit. binomial needs "
                        "read counts BiPCA cannot get from this stage, so it will fail")
    p.add_argument("--random_seed", type=int, required=True, help="random seed")
    p.add_argument("--n_iter", type=int, default=2000, help="Sinkhorn iteration cap")
    p.add_argument("--sinkhorn_tol", type=float, default=1e-5, help="Sinkhorn tolerance")
    p.add_argument("--min_gene_cells", type=int, default=10,
                   help="drop genes expressed in fewer cells than this. BiPCA's Sinkhorn "
                        "divides by squared column scalings and emits NaN on near-empty "
                        "columns; 50 was needed at n=14000 cells. Raise it if the run "
                        "aborts on NaN")
    # ponytail: no --n_components. mp_rank IS the output; k_max only bounds the
    # SVD, and saturating it aborts rather than publishing the ceiling as a rank.
    p.add_argument("--k_max", type=int, default=200,
                   help="singular vectors to compute. NOT a target: k is still mp_rank. "
                        "The run fails if mp_rank saturates this")
    return p.parse_args()


def _np(x):
    """BiPCA hands back torch tensors under its default backend."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def read_counts(h5ad_path, cell_ids, gene_ids):
    """Raw counts for the given cells x genes, in the order given."""
    a = ad.read_h5ad(h5ad_path)
    # DATA writes counts to layers['counts'] and leaves X empty (be1, tm-facs,
    # tenx all do); fall back to X for anything that does not.
    X = a.layers["counts"] if "counts" in a.layers else a.X
    if X is None:
        sys.exit(f"error: {h5ad_path} has neither layers['counts'] nor X")
    missing = set(cell_ids) - set(a.obs_names)
    if missing:
        sys.exit(f"error: {len(missing)} filtered cell ids absent from the h5ad, "
                 f"e.g. {sorted(missing)[:3]}")
    missing = set(gene_ids) - set(a.var_names)
    if missing:
        sys.exit(f"error: {len(missing)} selected gene ids absent from the h5ad, "
                 f"e.g. {sorted(missing)[:3]}")
    ci = a.obs_names.get_indexer(cell_ids)
    gi = a.var_names.get_indexer(gene_ids)
    X = sp.csr_matrix(X)[ci][:, gi]
    return X.astype(np.float32)


def read_tenx_genes(path):
    with h5py.File(path, "r") as h5:
        return [g.decode() for g in h5["matrix/genes"][:]]


def write_tsv(path, matrix, row_ids, row_label):
    cols = [f"PC{i + 1}" for i in range(matrix.shape[1])]
    with open(path, "w") as f:
        f.write(row_label + "\t" + "\t".join(cols) + "\n")
        for rid, row in zip(row_ids, matrix):
            f.write(rid + "\t" + "\t".join(f"{v:.10g}" for v in row) + "\n")


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.filtered_cellids, "rt") as f:
        cell_ids = [ln.strip() for ln in f if ln.strip()]
    gene_ids = read_tenx_genes(args.normalized_selected_h5)
    X = read_counts(args.rawdata_h5ad, cell_ids, gene_ids)
    print(f"  counts (cells x genes): {X.shape}")

    # Sinkhorn needs every row and column to carry mass. Genes go by the
    # --min_gene_cells guard, cells only if they are empty afterwards; both are
    # reported because a dropped cell is a cell missing from every downstream join.
    keep_g = np.asarray((X > 0).sum(axis=0)).ravel() >= args.min_gene_cells
    X, gene_ids = X[:, keep_g], [g for g, k in zip(gene_ids, keep_g) if k]
    keep_c = np.asarray(X.sum(axis=1)).ravel() > 0
    if not keep_c.all():
        print(f"  WARNING: dropping {int((~keep_c).sum())} cells with zero counts over "
              f"the surviving {X.shape[1]} genes; they will be absent downstream")
    X, cell_ids = X[keep_c], [c for c, k in zip(cell_ids, keep_c) if k]
    print(f"  after filtering: {X.shape}  (genes dropped: {int((~keep_g).sum())})")
    if min(X.shape) < 2:
        sys.exit("error: nothing left to factorize after filtering")

    from bipca import BiPCA  # imported late: it pulls in torch

    bp = BiPCA(variance_estimator=args.variance_estimator, seed=args.random_seed,
               n_iter=args.n_iter, sinkhorn_tol=args.sinkhorn_tol,
               n_components=args.k_max, verbose=1)
    bp.fit(X)

    k = int(bp.mp_rank)
    if k >= args.k_max:
        sys.exit(f"error: mp_rank={k} saturated --k_max {args.k_max}.\n"
                 f"       The reported rank would be the ceiling, not the MP edge. "
                 f"Raise --k_max.")
    scores = _np(bp.transform(counts=False, which="left"))   # U[:, :k] * shrunk S
    loadings = _np(bp.V_Y)[:, :k]
    # kst is only bound on some fit paths; q/sigma always are.
    diag = {"mp_rank": k, "q": float(bp.q), "sigma": float(np.ravel(bp.sigma)[0]),
            "ks": None if getattr(bp, "kst", None) is None else float(np.ravel(bp.kst)[0]),
            "n_cells": len(cell_ids), "n_genes": len(gene_ids),
            "variance_estimator": args.variance_estimator,
            "min_gene_cells": args.min_gene_cells, "random_seed": args.random_seed}
    print(f"  mp_rank={k} q={diag['q']:.4g} sigma={diag['sigma']:.4g} KS={diag['ks']} "
          f"scores={scores.shape} loadings={loadings.shape}")

    # NaN is BiPCA's failure mode, not an exception: near-empty columns make the
    # squared column scaling blow up and the NaN propagates through the SVD.
    for what, M in (("scores", scores), ("loadings", loadings)):
        if not np.isfinite(M).all():
            sys.exit(f"error: {what} contain non-finite values -- Sinkhorn diverged on "
                     f"sparse columns. Raise --min_gene_cells above {args.min_gene_cells}.")
    if scores.shape != (len(cell_ids), k) or loadings.shape != (len(gene_ids), k):
        sys.exit(f"error: shape mismatch -- scores {scores.shape} / loadings "
                 f"{loadings.shape} vs {len(cell_ids)} cells x {len(gene_ids)} genes, k={k}")

    write_tsv(out / f"{args.name}_embedding.tsv", scores, cell_ids, "cell_id")
    write_tsv(out / f"{args.name}_loadings.tsv", loadings, gene_ids, "gene_id")
    (out / f"{args.name}_bipca.json").write_text(json.dumps(diag, indent=2))
    print(f"  wrote: {out}/{args.name}_{{embedding,loadings}}.tsv + _bipca.json")


if __name__ == "__main__":
    main()
