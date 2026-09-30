#!/usr/bin/env python3
"""End-to-end smoke test: synthesise the three RDIMR inputs (raw-count h5ad,
gzipped cell and feature ids), run pca.py, check the output
contract. Run from the module root: python tests/smoke.py
"""

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

N_CELLS, N_GENES, N_SEL, K = 300, 400, 200, 4
rng = np.random.default_rng(0)

# Low-rank Poisson counts: K latent programs, so the MP edge has something to find.
W = rng.gamma(2.0, 1.0, size=(N_CELLS, K))
H = rng.gamma(2.0, 1.0, size=(K, N_GENES))
counts = rng.poisson(W @ H).astype(np.float32)
cells = [f"cell{i}" for i in range(N_CELLS)]
genes = [f"gene{j}" for j in range(N_GENES)]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    a = ad.AnnData(shape=(N_CELLS, N_GENES))
    a.obs_names, a.var_names = cells, genes
    a.layers["counts"] = sp.csr_matrix(counts)  # DATA leaves X empty
    a.write_h5ad(tmp / "t.h5ad")

    # FILT keeps a subset of cells and a subset of genes.
    kept_cells, kept_genes = cells[:-20], genes[:N_SEL]
    for name, ids in (("cellids", kept_cells), ("featureids", kept_genes)):
        with gzip.open(tmp / f"{name}.txt.gz", "wt") as f:
            f.write("\n".join(ids) + "\n")

    r = subprocess.run([sys.executable, "pca.py", "--output_dir", str(tmp),
                        "--name", "t", "--rawdata_h5ad", str(tmp / "t.h5ad"),
                        "--filtered_cellids", str(tmp / "cellids.txt.gz"),
                        "--filtered_featureids", str(tmp / "featureids.txt.gz"),
                        "--random_seed", "42", "--min_gene_cells", "5"])
    assert r.returncode == 0, "pca.py failed"

    pcas = pd.read_csv(tmp / "t_embedding.tsv", sep="\t", index_col=0)
    load = pd.read_csv(tmp / "t_loadings.tsv", sep="\t", index_col=0)
    diag = json.loads((tmp / "t_bipca.json").read_text())

    # Contract: the cells FILT kept (minus any zero-count drops), the genes FILT
    # kept, k columns on both sides, and k is what BiPCA reported.
    assert set(pcas.index) <= set(kept_cells), "leaked cells FILT dropped"
    assert len(pcas) == diag["n_cells"] and len(load) == diag["n_genes"]
    assert set(load.index) <= set(kept_genes), "leaked genes FILT dropped"
    assert list(pcas.columns) == list(load.columns) == [f"PC{i+1}" for i in range(diag["mp_rank"])]
    assert np.isfinite(pcas.values).all() and np.isfinite(load.values).all()
    assert diag["mp_rank"] >= 1, "MP truncation found no signal in a rank-4 matrix"

    # the guard: non-integer "counts" must be refused, not fitted
    b = ad.AnnData(shape=(N_CELLS, N_GENES))
    b.obs_names, b.var_names = cells, genes
    b.layers["counts"] = sp.csr_matrix(np.log1p(counts))
    b.write_h5ad(tmp / "bad.h5ad")
    r = subprocess.run([sys.executable, "pca.py", "--output_dir", str(tmp),
                        "--name", "bad", "--rawdata_h5ad", str(tmp / "bad.h5ad"),
                        "--filtered_cellids", str(tmp / "cellids.txt.gz"),
                        "--filtered_featureids", str(tmp / "featureids.txt.gz"),
                        "--random_seed", "42"], capture_output=True, text=True)
    assert r.returncode != 0 and "raw counts" in r.stderr, r.stderr
    # ablations (lattice W5/W6): N past mp_rank, three column scalings
    def run(name, *extra):
        r = subprocess.run([sys.executable, "pca.py", "--output_dir", str(tmp), "--name", name,
                            "--rawdata_h5ad", str(tmp / "t.h5ad"),
                            "--filtered_cellids", str(tmp / "cellids.txt.gz"),
                            "--filtered_featureids", str(tmp / "featureids.txt.gz"),
                            "--random_seed", "42", "--min_gene_cells", "5", *extra],
                           capture_output=True, text=True)
        return r, (pd.read_csv(tmp / f"{name}_embedding.tsv", sep="\t", index_col=0).to_numpy()
                   if r.returncode == 0 else None)
    k, n = diag["mp_rank"], diag["mp_rank"] + 3
    for scaling in ("shrunk", "raw", "none"):
        r, E = run(scaling, "--ablate_n_components", str(n), "--component_scaling", scaling)
        assert r.returncode == 0, r.stderr
        d = json.loads((tmp / f"{scaling}_bipca.json").read_text())
        assert E.shape == (diag["n_cells"], n) and d["mp_rank"] == k and d["n_components_used"] == n
        norms = np.linalg.norm(E, axis=0)
        if scaling == "shrunk":   # the shrinker zeroes everything past the MP edge
            assert (norms[k:] == 0).all() and (norms[:k] > 0).all(), norms
        elif scaling == "raw":    # unshrunk S: descending, nothing zeroed
            assert (norms > 0).all() and (np.diff(norms) <= 1e-6 * norms[0]).all(), norms
        else:
            assert np.allclose(norms, 1.0), norms
    r, _ = run("toobig", "--ablate_n_components", "201")   # --k_max defaults to 200
    assert r.returncode != 0 and "do not clamp" in r.stderr, r.stderr

    print(f"OK: {pcas.shape[0]} cells x {diag['mp_rank']} PCs, "
          f"{load.shape[0]} genes, q={diag['q']:.3f}")
