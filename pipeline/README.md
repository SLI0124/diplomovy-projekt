# Pipeline — Quick Start & Usage 🚀

A compact guide to running the `pipeline` module. This pipeline *must* be started via `main.py` (no other activation mechanisms are supported).

---

## What this does 🔧

- Downloads raw data (consumption, weather, price).
- Processes data (datetime features, consumption, weather, price).
- Merges processed outputs into the final combined dataset.

Everything is coordinated by `main.py` which is the single supported entry point.

---

## Where to run it 📁

You should run the pipeline from the `pipeline` directory.

```bash
cd pipeline
python main.py --help
```

Do **not** run the pipeline from the repository root — paths and relative behavior assume you started inside `pipeline/`.

---

## Basic usage & flags ✅

- Show help:

```bash
python main.py --help
```

- Run the full pipeline (download + process everything, **recommended**):

```bash
python main.py --all
```

- Download only consumption (for selected networks):

```bash
python main.py --download consumption --consumption-networks gasnet vcpnet
```

- Process only the merge step up to a specific end date:

```bash
python main.py --process merge --end-date 2024-12-31
```

Flags summary:

- `--download`: `consumption` | `weather` | `price` | `all`
- `--process`: `dates` | `consumption` | `weather` | `price` | `merge` | `all`
- `--all`: Shorthand to download and process everything
- `--end-date`: End date in `YYYY-MM-DD` (defaults to last day of the previous month)
- `--consumption-networks`: Space-separated network keys (defaults to: `gasnet`, `vcpnet`, `jmpnet`, `smpnet`).

---

## Date defaults & limits 📅

- The pipeline's common start date is **2013-01-01**. Consumption downloads start one day earlier to cover full windows. This can be overridden by specifying changing value `COMMON_START_DATE` in `config.py`
- If `--end-date` is omitted, the pipeline uses the **last day of the previous month** as the end date. (E.g., if today is 2024-06-15, the default end date is 2024-05-31.)

> **HTTP reliability note:** Consumption downloads now send browser-like request headers (including `User-Agent`/`Accept`) and automatically retry transient failures (`403`, `429`, `5xx`) with exponential backoff. This significantly reduced unauthorized/blocked request issues, but occasional upstream failures may still happen.
