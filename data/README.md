# Data

Raw and processed dataset files are intentionally **not** committed (see `.gitignore`).

- `raw/` — original downloaded files per dataset (ETTh1/h2/m1/m2, Electricity, Traffic,
  Weather, PEMS03/04/07/08, M4). Populate via a future `scripts/download_data.py`.
- `processed/` — tokenized/cached tensors derived from `raw/`.

Dataset specs (dimensions, split ratios, frequency) are in Table 1 of the paper —
these will be encoded as per-dataset configs under `configs/` once the data
pipeline is implemented.
