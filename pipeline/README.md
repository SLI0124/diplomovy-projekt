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

- Download PPNet consumption (experimental / opt-in):

```bash
python main.py --download consumption --consumption-networks ppnet --include-ppnet
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
- `--include-ppnet`: Explicit opt-in for `ppnet` consumption (experimental / occasionally faulty).

---

## Date defaults & limits 📅

- The pipeline's common start date is **2013-01-01**. Consumption downloads start one day earlier to cover full windows. This can be overridden by specifying changing value `COMMON_START_DATE` in `config.py`
- If `--end-date` is omitted, the pipeline uses the **last day of the previous month** as the end date. (E.g., if today is 2024-06-15, the default end date is 2024-05-31.)
- Network-specific apply: **PPNET** data is not available before **2016-01-01** by default (see PPNet section below).

---

## PPNet legacy data ⚠️

PPNet support has had a lot of work put into it, but the upstream/source data can be
unreliable, so it is **opt-in**. By default, the pipeline downloads and processes only
`gasnet`, `vcpnet`, `jmpnet`, `smpnet`.

To include PPNet in downloads/processing, use `--include-ppnet`.

### `--include-ppnet` (where it applies)

This flag only affects **consumption**.

It is respected when you run:

- `--all` (affects the consumption download + consumption processing steps)
- `--download all` (affects only the consumption part)
- `--download consumption`
- `--process consumption`
- `--process merge` (passes the consumption network list into the merge step for validation)

It does **nothing** if you run only `--download weather`, `--download price`, `--process weather`, or `--process price`.

### `--include-ppnet` (behavior rules)

- If you do **not** use `--include-ppnet`, the defaults are: `gasnet`, `vcpnet`, `jmpnet`, `smpnet`.
- If you use `--include-ppnet` **without** `--consumption-networks`, the pipeline will run consumption with:
  `gasnet`, `vcpnet`, `jmpnet`, `smpnet`, `ppnet`.
- If you specify `--consumption-networks ... ppnet ...` without `--include-ppnet`, the CLI will error.

### Examples

- Full pipeline including PPNet:

```bash
python main.py --all --include-ppnet
```

- Download consumption defaults + PPNet (no need to list networks):

```bash
python main.py --download consumption --include-ppnet
```

- PPNet only:

```bash
python main.py --download consumption --consumption-networks ppnet --include-ppnet
python main.py --process consumption --consumption-networks ppnet --include-ppnet
```

- There is a helper script to generate PPNet-style daily CSVs from a legacy personal file:

```bash
cd tools
python ppnet_data_extractor.py
```

- The extractor's defaults cover **2013-01-01 → 2015-12-31** and it expects an input file (default `data/input_ppnet_consumption.csv`).
- I do **not** distribute this original legacy file. If you need the 2013–2015 raw input, **you can contact me** and I can share it (subject to data/legal constraints). Otherwise you will either have empty PPNet input for that period or you will be limited to PPNet data from 2016 onwards (pipeline enforces a PPNet minimum date of 2016-01-01).

> **HTTP reliability note:** Consumption downloads now send browser-like request headers (including `User-Agent`/`Accept`) and automatically retry transient failures (`403`, `429`, `5xx`) with exponential backoff. This significantly reduced unauthorized/blocked request issues, but occasional upstream failures may still happen.
