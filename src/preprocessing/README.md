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
python .\main.py
```

Optional:

```bash
python .\main.py --input <merged.csv> --output <cleaned.csv> --report <report.json>
```

## Report 📊

The JSON report is compact and contains:

- `meta`: row count + time span,
- `config`: thresholds and tree params,
- `mask_counts`: how many values were invalidated,
- `imputation`: per-column tree/fallback fill stats,
- `checks`: remaining NaN count, rule violations, daily price consistency, recomputed total rows.

Console output also prints step-by-step progress (`[1/5] ... [5/5]`).

## References 📚

- [Prague's Smetana Embankment Closes Gas Leak for Three Hours](https://www.irozhlas.cz/regiony/prazske-smetanovo-nabrezi-na-tri-hodiny-uzavrel-unik-plynu_201602232030_mtaborska) - real-world event reflected in data quality issues, not an outlier but a known anomaly.
