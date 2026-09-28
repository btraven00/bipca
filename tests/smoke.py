#!/usr/bin/env python3
"""End-to-end smoke test: synthesise the three RDIMR inputs (raw-count h5ad,
gzipped cell ids, TENx h5 of selected genes), run pca.py, check the output
contract. Run from the module root: python tests/smoke.py
"""

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import h5py
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

    # FILT keeps a subset of cells; FEAT keeps a subset of genes.
    kept_cells, kept_genes = cells[:-20], genes[:N_SEL]
    with gzip.open(tmp / "cellids.txt.gz", "wt") as f:
        f.write("\n".join(kept_cells) + "\n")
    sel = sp.csc_matrix(counts[:, :N_SEL].T)  # TENx is genes x cells
    with h5py.File(tmp / "sel.h5", "w") as h5:
        g = h5.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = sel.data, sel.indices, sel.indptr
        g["shape"] = np.array(sel.shape)
        g["genes"] = np.array([s.encode() for s in kept_genes])
        g["barcodes"] = np.array([s.encode() for s in cells])

    r = subprocess.run([sys.executable, "pca.py", "--output_dir", str(tmp),
                        "--name", "t", "--rawdata_h5ad", str(tmp / "t.h5ad"),
                        "--filtered_cellids", str(tmp / "cellids.txt.gz"),
                        "--normalized_selected_h5", str(tmp / "sel.h5"),
                        "--random_seed", "42", "--min_gene_cells", "5"])
    assert r.returncode == 0, "pca.py failed"

    pcas = pd.read_csv(tmp / "t_embedding.tsv", sep="\t", index_col=0)
    load = pd.read_csv(tmp / "t_loadings.tsv", sep="\t", index_col=0)
    diag = json.loads((tmp / "t_bipca.json").read_text())

    # Contract: the cells FILT kept (minus any zero-count drops), the genes FEAT
    # selected, k columns on both sides, and k is what BiPCA reported.
    assert set(pcas.index) <= set(kept_cells), "leaked cells FILT dropped"
    assert len(pcas) == diag["n_cells"] and len(load) == diag["n_genes"]
    assert set(load.index) <= set(kept_genes), "leaked genes FEAT dropped"
    assert list(pcas.columns) == list(load.columns) == [f"PC{i+1}" for i in range(diag["mp_rank"])]
    assert np.isfinite(pcas.values).all() and np.isfinite(load.values).all()
    assert diag["mp_rank"] >= 1, "MP truncation found no signal in a rank-4 matrix"
    print(f"OK: {pcas.shape[0]} cells x {diag['mp_rank']} PCs, "
          f"{load.shape[0]} genes, q={diag['q']:.3f}")
