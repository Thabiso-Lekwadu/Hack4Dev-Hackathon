# Data

The challenge data is not stored in this repository (about 670 MB once features are built).

Place these two files here before running the notebooks:

- `madrid_train.parquet`
- `amsterdam_data.parquet`

Notebook 5 writes the intermediate cube to `data/cache/` and the engineered features to `data/features/`; both are recreated on each run.
