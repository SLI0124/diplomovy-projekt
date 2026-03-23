# Preprocessing  ⚙️

Compact summary of the current preprocessing implementation for the merged gas dataset.

Analysis background is in [this notebook](../../notebooks/0_preprocssing_eda.ipynb).

## Scope 🎯

- Clean **consumption** and **price** columns with fixed rules.
- Handle **weather only for NaN** values.
- Keep timeline continuous (mask invalid values to NaN, then impute).
- Treat price as **daily value replicated to 24 hours**.
- Recompute `consumption_total` only when component values changed (or original total was invalid).

## Steps I used 🧩

1. Build hourly timestamp index from `year/month/day/hour`.
2. Mask invalid values in consumption, total, and price columns.
3. Impute hourly consumption/weather with **DecisionTreeRegressor** + month-hour fallback.
4. Impute **price on daily level**, enforce price constraints, then broadcast back to hourly rows.
5. Recompute `consumption_total` only on affected rows.
6. Build a compact QA report with key validity checks.

## Config values 🛠️

Defined once in `value_cleaning.py` via `PreprocessingConfig` defaults:

- `consumption_upper_bound = 800000`
- `max_price_upper_bound = 500`
- `tree_max_depth = 14`
- `tree_min_samples_leaf = 20`
- `tree_min_train_rows = 300`
- `tree_random_state = 42`

Runner (`main.py`) currently uses these defaults directly.

## Important rules implemented ✅

- **Consumption components** (`gasnet`, `jmpnet`, `smpnet`, `vcpnet`): invalid if missing, `<= 0`, or `> 800000`.
- **Consumption total**: invalid if missing or `<= 0`; recomputed where needed.
- **Price**: invalid if missing or `== 0`; additionally `max_price_eur_mwh > 500` is invalid.
- **Price consistency**: enforce daily constraints and keep one value/day (replicated across 24 hours).
- **Weather**: no outlier filtering, only NaN filling.

## Run 🚀

Run from `src/preprocessing`:

```bash
python main.py
```

Optional:

```bash
python main.py --input <merged.csv> --output <cleaned.csv> --report <report.json>
```

Print full JSON report to console only when needed:

```bash
python main.py --print-report
```

## New CLI features 🆕

Default behavior is unchanged:

- `--output` still saves the original cleaned dataset.
- `--report` still saves the preprocessing JSON report.
- New features are opt-in via flags.

Additional dataset exports are saved to a dedicated subfolder under output directory:

- default subfolder: `splits`
- nested layout to avoid confusion/overwrites between file groups:
  - `splits/<variant_stem>/`
  - `splits/<variant_stem>/ranges_from_<anchor_year>_to_<max_year>/`
  - `splits/<variant_stem>/single_years/`
  - `splits/<variant_stem>/run_params.json` (machine-readable run manifest)

### Readable parameter-based variant naming

- Variant stem is built directly from enabled parameters (no hash), for example:
  - `cyc-hour-month-day-of-week-src-dropped__lag-consumption-total-1-24-168`
- If no feature flags are used, variant stem is `base`.
- Split file names are short:
  - cumulative ranges: `range_<start>_<end>.csv`
  - single years: `year_<year>.csv`

### Export options

- Cumulative ranges and single-year exports are always generated automatically.
- Range anchor/start year is fixed to `2013`.
- `--exports-subdir` : default `splits`

### Feature engineering options

- Drop columns:
  - `--drop-columns year,holiday,before_holiday`
- Cyclical datetime features:
  - `--add-cyclical`
  - `--cyclical-columns hour,month,day_of_week` (default)
  - `--drop-cyclical-source-columns`
- Lag features:
  - `--add-lag-features`
  - `--lag-columns consumption_total` (default)
  - `--lag-counts 1,24,168` (default)
- Rolling features:
  - `--add-rolling-features`
  - `--rolling-columns consumption_total` (default)
  - `--rolling-windows 24,168` (default)
  - `--rolling-aggregation mean|sum|both` (default `mean`)
- Expanding features:
  - `--add-expanding-features`
  - `--expanding-columns consumption_total` (default)
  - `--expanding-min-periods 24` (default)
  - `--expanding-aggregation mean|sum|both` (default `mean`)

### Full-range generation example (all requested files)

```bash
python main.py --add-cyclical --drop-cyclical-source-columns --add-lag-features --add-rolling-features --rolling-aggregation both --add-expanding-features --expanding-aggregation both
```

This run keeps the original cleaned file in `--output` and generates all cumulative ranges plus single-year files under a readable parameter-based folder in `splits`.

Note: cumulative ranges intentionally stop one year before the latest year in data, so the latest year remains available as a dedicated holdout/test year.

### Usage examples

- Cyclical features + drop cyclical source columns + all splits:

```bash
python main.py --add-cyclical --drop-cyclical-source-columns
```

- Cyclical features + drop non-total consumption components + all splits:

```bash
python main.py --add-cyclical --drop-cyclical-source-columns --drop-columns consumption_gasnet,consumption_jmpnet,consumption_smpnet,consumption_vcpnet
```

- Drop only non-total consumption components (no extra temporal features):

```bash
python main.py --drop-columns consumption_gasnet,consumption_jmpnet,consumption_smpnet,consumption_vcpnet
```

- Cyclical + lag features (no rolling/expanding) + all splits:

```bash
python main.py --add-cyclical --drop-cyclical-source-columns --add-lag-features
```

## Report 📊

The JSON report is compact and contains:

- `meta`: row count + time span,
- `config`: thresholds and tree params,
- `mask_counts`: how many values were invalidated,
- `imputation`: per-column tree/fallback fill stats,
- `checks`: remaining NaN count, rule violations, daily price consistency, recomputed total rows.

Console output prints step-by-step progress (`[1/5] ... [5/5]`) and summary paths/counts.
Full JSON report is quiet by default and printed only with `--print-report`.

## References 📚

- [Prague's Smetana Embankment Closes Gas Leak for Three Hours](https://www.irozhlas.cz/regiony/prazske-smetanovo-nabrezi-na-tri-hodiny-uzavrel-unik-plynu_201602232030_mtaborska) - real-world event reflected in data quality issues, not an outlier but a known anomaly.
