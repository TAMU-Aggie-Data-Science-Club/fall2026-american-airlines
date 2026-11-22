# Team 03 technical stack — proposed adoption
Initial standard is **Python 3.11, venv/pip**, VS Code and Jupyter; **pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, Matplotlib/Seaborn, pytest, Ruff**, GitHub private Issues/PRs and synthetic-only Actions. Frozen compatible dependency pins are an acceptance condition, not claimed complete.

Core design: plain Python reusable package (`team3_forecast/`) with ingestion/validation, date and availability contract, feature builder, seasonal baseline, optional classical models, pooled ML challenger, rolling-origin backtest, paired scoring, experiment runner, registry and report generator.

Optional with evidence:
- **DuckDB + PyArrow/Parquet:** local joins/table storage as data size or complexity justifies.
- **Local MLflow + SQLite:** richer experiment tracking after stable immutable manifests; no external tracking URI.
- **Optuna:** bounded validation-only hyperparameter search after baseline/backtest is trustworthy.
- **Plotly/Streamlit:** approved local-only dashboard if time allows.
- **Databricks/Spark/Delta:** not required; sponsor-approved workspace only after documented need, permission, owner and budget.
- **PyTorch/neural models:** research option only, never the core release condition.

Every environment and dependency should be pinned and tested on clean machines. Real-data training occurs exclusively where permitted; cloud CI tests only synthetic fixtures. This `docs/` proposal does not authorize any installation or external data movement.

## Verified integration environment
The November 21 integration is tested locally on Python 3.13 with requirements-lock.txt. The existing package name is aaforecast. A synthetic-only CI workflow is defined; remote execution must be checked separately. Public airport reference, bounded local ablations and run manifests are implemented; cloud platforms remain optional.
