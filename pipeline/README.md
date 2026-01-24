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
python main.py --download consumption --consumption-networks gasnet ppnet
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
- `--consumption-networks`: Space-separated network keys (defaults to all). Supported keys: `gasnet`, `vcpnet`, `jmpnet`, `smpnet`, `ppnet`.

---

## Date defaults & limits 📅

- The pipeline's common start date is **2013-01-01**. Consumption downloads start one day earlier to cover full windows. This can be overridden by specifying changing value `COMMON_START_DATE` in `config.py`
- If `--end-date` is omitted, the pipeline uses the **last day of the previous month** as the end date. (E.g., if today is 2024-06-15, the default end date is 2024-05-31.)
- Network-specific apply: **PPNET** data is not available before **2016-01-01** by default (see PPNet section below).

---

## PPNet legacy data ⚠️

- There is a helper script to generate PPNet-style daily CSVs from a legacy personal file:

```bash
cd tools
python ppnet_data_extractor.py
```

- The extractor's defaults cover **2013-01-01 → 2015-12-31** and it expects an input file (default `data/input_ppnet_consumption.csv`).
- I do **not** distribute this original legacy file. If you need the 2013–2015 raw input, **you can contact me** and I can share it (subject to data/legal constraints). Otherwise you will either have empty PPNet input for that period or you will be limited to PPNet data from 2016 onwards (pipeline enforces a PPNet minimum date of 2016-01-01).

> **Heads-up:** Consumption downloads have occasionally returned HTTP `403` on some dates. In my experience a simple retry (re-running the download for the same date/network or re-running `--all`) usually succeeds.
