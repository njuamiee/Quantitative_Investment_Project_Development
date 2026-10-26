# 2026-10-26 exhaustive research

See `系统重设计_20261026/研究协议.md` for the frozen research design and its limitations.

For a Chinese beginner's explanation, read `系统重设计_20261026/先读这份说明.md` and the live HTML report. The report contains 18 charts, including paired comparisons for every declared model coordinate and every account-rule coordinate. These are descriptive comparisons, not independent trials or causal population estimates. The monthly decision details distinguish prediction horizon, model refitting frequency and account rebalancing interval.

Use the existing research Python environment, from the repository root:

```powershell
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/prepare.py
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/features.py
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/test_system.py
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/runner.py --workers 6
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/minute_study.py
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/build_report.py
.venv_research_20261011/Scripts/python.exe tools/systematic_20261026/audit.py
```

`runner.py` registers 8,640 model configurations and 27,993,600 distinct parameter configurations after removing exact h=1 rebalance duplicates. These are declared counts, never completion claims. Predictions and actual fit counts are recorded separately. Every finished model configuration runs its entire declared risk grid. `tasks.sqlite3`, `progress.json` and per-configuration `account_complete.json` are authoritative execution receipts.

`--limit N` runs the next N pending model configurations for engineering checks; those results are still identified as a partial batch. Re-run the command to resume after interruption. Do not start two runners against the same queue. The parent uses six local worker processes (ordinary Python computation, not delegated AI agents), one BLAS thread each. The Windows background launcher uses hidden windows. Keep the machine awake for computation to continue; this project does not change OS power settings.

`watch_report.py` refreshes the HTML and PNG chart snapshot every five minutes when completed counts change. It exits after the queue reaches a terminal state and writes a final audit receipt. It does not notify external services or place trades. A stale heartbeat is visibly reported instead of being called success.

Distributed continuation uses disjoint shards: `core` is the 960 daily-library/252-session configurations; `xgboost_extra` is the remaining 960 XGBoost configurations, kept on Windows. `remote` is the 6,720 non-core, non-XGBoost configurations on the existing data server, in a task-owned directory under `/tmp`, with 40 one-thread workers. `local_next.py` waits for the core shard before starting the Windows XGBoost expansion. `server_control.py watch` transfers only completed artifacts, checks archive, prediction and results SHA256 values, then publishes the completion receipt locally. It never uploads credentials or modifies the original server data directory.

The eight-family reference check found identical ranks for seven families (Ridge floating-point score difference up to 1.53e-7); XGBoost differed substantially between the Windows build and Linux CPU package at the same version. Accordingly all XGBoost performance cases use Windows, not a mix of these builds. Reference checks and unit tests are additional engineering runs, not included in the main grid's retained-fit count.

`minute_study.py` runs 1,152 separately registered 2026Q3 minute-account cases with frozen Q4 minute-model selection. Its coverage and execution assumptions differ from the daily grid and it is not silently pooled into the daily-grid count.

Caches are valid only for this frozen source and input vintage. If model preparation, screening, training or ledger semantics change, use a new RUN directory/revision; never reuse old `complete.json` markers under changed semantics. Pure report wording/formatting changes may reuse computed accounts. No raw supplier files, credentials, fitted per-date binary models or prediction caches need to be committed. Results use actual source data; no generated price series enter any reported investment performance.

Main dependencies are the existing observed research lock plus `numba==0.68.0` and `llvmlite==0.50.0`. The tools require local raw datasets described by earlier project input manifests. Clone alone is insufficient without the authorized data.
