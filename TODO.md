# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] add price column downloader and processor, take endpoint from `price_downloader.cpp`
- [ ] update **jupyter notebook** with new data sources, new columns, new logic, new everything
- [ ] omg I found `ppnet` data source for consumption, I need to implement it instead of gasnet or perhaps add a parameter to choose which one to use, I could also download them all  
  - [ ] this would also mean I need to check main merging logic to avoid naming misconceptions

## Next steps

- [ ] I think I am creating *before_holiday* correctly, but in original dataset it seems that it is *actually* after holiday, I should keep that in mind or if I am creating probably a new version of it, I can do whatever the fuck I want
- [ ] I've set default start date for all scripts to **2013.1.1**, right now I deleted param for start date, I think it won't be changed at all but it'd be nice to have a way to change it somehow, perhaps I can do like config file with dirs, start date, etc.
- [ ] I have some duplicate code, create `utils.py` with all functions used across scripts
- [ ] maybe check *float* and *int* values in final dataset, if one can be one or the other
- [ ] look for some other potential data source for new features (optional)
- [ ] type checking
- [ ] better docstrings with params, return types, etc.
- [ ] README file for `pipeline` directory
- [ ] for processors try checking if `raw` files are even existing, if not, throw error or warning, DO NOT call downloader from processor
- [ ] set up some black, pylint or other code formatting and linting tools for repository project

## Columns implementation status

- [X] year - will be generated quite easily
- [X] month - will be generated quite easily
- [X] day - will be generated quite easily
- [X] hour - will be generated quite easily
- [X] day_of_week - will be generated quite easily
- [X] before_holiday - need to ask, will be scraped
- [X] holiday - need to ask, will be scraped, it does not make much sense to type them out manually
- [X] consumption - gasnet není dostupný, odkazy v `consumption_downloader.cpp`
- [X] temperature - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] pressure - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] pressure2 - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] humidity - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] wind_direction - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] wind_speed - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] max_gust - `data_combiner.cpp` is loading file named `input_ppnet_weather.csv` and parsing MISSING and EMPTY values
- [X] phenomena - same as above, probably
- [X] recent_phenomena - same as above, probably
- [X] cloud_cover - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] visibility [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [X] dewpoint - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] quantity — traded volume / amount
- [ ] price — settlement/trade price

some new columns were added or replaced, need to consult about them

### Notes for consultation

- `price` is in PDF but not in actual dataset csv, it is also in code he provided
- `consumption` is not available via API, gasnet probably change their endpoints, I will have to ask them directly, old links are in `consumption_downloader.cpp`

---

- I can't find ppnet, which seems to be source of consumption data, gasnet still provides `gasnet`, `vcpnet`, `jmpnet` and `smpnet` datasets, but not ppnet
  - this means it is not possible to recreate consumption data exactly as it was in original dataset
- also getting data from russian site is very tricky and requires higher level of scraping via selenium, so perhaps I should look for better source of weather data, so recreating original dataset is not priority if I can't get consumption data exactly as it was
  - this also means that columns are not gonna be exactly the same as in original dataset
    - that might be also a good thing, since I can create better dataset than original one, with more relevant columns and better data sources
- in new dataset for cloud coverage there is no text whatsoever, only numerical values, new welcomed additions
  - once again, it won't be exactly the same as original dataset

### Notes after consultation

- firstly recreate the dataset as it is now, no new column needed
  - look into new source of **gasnet** data
- secondly try to find better source than that russian one for weather data, they are probably stealing it from somewhere else so I should be able to find it as well
- he said there is algorithm for calculating easter holiday, I should look into it
- for now, keep the before holiday and holiday as it is, we can resolve it later
- he was searching for some prazska plynarenska and chmi, might give it a look

---

## Data Quality Analysis Summary

### Critical Data Issues ✅ RESOLVED

**Problem**: 100% missing consumption data due to data type mismatch and temporal offset in gasnet files.

**Solution**: Complete consumption processor rewrite with cross-file temporal alignment, proper data types, and NA value handling.

### Results Summary

**Overall Success**: 112,393 merged records with 99.86% consumption data availability

| Year | Expected | Merged | Missing Consumption | Status |
|------|----------|--------|-------------------|--------|
| 2013 | 8760     | 8760   | **0** ✅         | Perfect |
| 2014 | 8760     | 8760   | **0** ✅         | Perfect |
| 2015 | 8760     | 8760   | **0** ✅         | Perfect |
| 2016 | 8784     | 8784   | **21** (NA)      | Near-perfect |
| 2017 | 8760     | 8760   | **0** ✅         | Perfect |
| 2018 | 8760     | 8760   | **11** (NA)      | Near-perfect |
| 2019 | 8760     | 8760   | **13** (NA)      | Near-perfect |
| 2020 | 8784     | 8784   | **43** (NA)      | Near-perfect |
| 2021 | 8760     | 8760   | **26** (NA)      | Near-perfect |
| 2022 | 8760     | 8760   | **0** ✅         | Perfect |
| 2023 | 8760     | 8760   | **0** ✅         | Perfect |
| 2024 | 8784     | 8784   | **0** ✅         | Perfect |
| 2025 | 7201     | 7201   | **23** (NA)      | Work in progress |

| Status | Years | Missing Hours |
|--------|-------|---------------|
| ✅ Perfect (0 missing) | 2013-2015, 2017, 2022-2024 | 0 |
| 🟡 Near-perfect | 2016, 2018-2021 | 11-43 hours |
| 🔄 Work in progress | 2025 | 23 hours |

**Key Improvements**:

- Fixed data type compatibility (float→Int64)
- Implemented cross-file temporal alignment for gasnet's split-day structure
- Added malformed file detection with NA filling
- Switched from row concatenation to date-based joins

**Impact**: Transformed from 100% missing consumption data to 99%+ availability across all years.
