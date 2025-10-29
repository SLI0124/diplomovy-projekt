# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [X] look for columns that are easily and still obtainable via API endpoints for consultation
- [X] make clear boundaries of what is downloader, what is processor, what is final dataset assembler
- [X] add requirements for conda and pip, you can look generally for some environments and project settings overall
- [X] gas net for consumption data is having maintanance so I need to process them later (they have ratelimit)
- [ ] **CRITICAL**: Fix data inconsistencies in processed files - mismatched row counts between data sources
  - [ ] maybe for current year set the last possible day to last day of previous month instead of yesterday

## Next steps

- [x] any kind of column saving into `data/processed` folder or something like that
- [x] create a main downloader script that sequentially calls individual downloader modules; allow selective downloading (e.g., only weather data) or downloading all data at once
- [x] same logic for processors
- [ ] testing values from original dataset against newly acquired values for consistency
- [ ] I think I am creating *before_holiday* correctly, but in original dataset it seems that it is *actually* after holiday, I should keep that in mind
- [ ] at the complete end of pipeline, create final dataset assembler that merges all processed columns into one final dataset ready for modeling, taking care of any necessary alignment of latest dates across all columns or handling missing data appropriately

## Ideas

- since we probably won't be exceeding it into the past, only going forward, we can easily test acquired data for consistency and compare new data against old data, this way this project can be run in the future and it will be assured that the data is correct even after multiple months or years perhaps

## Columns implementation status

- [X] year - will be generated quite easily
- [X] month - will be generated quite easily
- [X] day - will be generated quite easily
- [X] hour - will be generated quite easily
- [X] day_of_week - will be generated quite easily
- [X] before_holiday - need to ask, will be scraped
- [X] holiday - need to ask, will be scraped, it does not make much sense to type them out manually
- [ ] consumption - gasnet není dostupný, odkazy v `consumption_downloader.cpp`
- [ ] temperature - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] pressure - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] pressure2 - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] humidity - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] wind_direction - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] wind_speed - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] max_gust - `data_combiner.cpp` is loading file named `input_ppnet_weather.csv` and parsing MISSING and EMPTY values
- [ ] phenomena - same as above, probably
- [ ] recent_phenomena - same as above, probably
- [ ] cloud_cover - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] visibility [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)
- [ ] dewpoint - [data source](http://rp5.ru/metar.php?metar=LKKB&lang=en)

### Notes for consultation

- `year`, `month`, `day`, `hour` and `day of week` will be handled purely in code, no need to download them from anywhere, I will look into some library that can do it for me, for example pandas or datetime, it will be probably handled in processors directory
  - since we don't know, what columns will have up to date data, this will be handled at the end and check the the most oldest date in the dataset and rest will be cut from that date forward
- `before_holiday` and `holiday` will be downloaded from web, I will try to find some API or I will have to scrape it from `https://www.kurzy.cz/kalendar/statni-svatky/{year}/`, this is also
  - also why the hell is column `before_holiday` actually after holiday? I should ask beforehand and soon to be perfectly clear
- `price` is in PDF but not in actual dataset csv, it is also in code he provided
- `consumption` is not available via API, gasnet probably change their endpoints, I will have to ask them directly, old links are in `consumption_downloader.cpp`
  - našel jsem odkaz [tady](https://www.gasnet.cz/dalsi-sluzby/pro-stavare-a-projektanty/zadost-o-vektorova-data)
- další webovka, kterou jsem našel pro počasí je [tady](https://mesonet.agron.iastate.edu/request/download.phtml?network=CZ__ASOS), ale nevím, jestli to není jenom pro placené uživatele

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

### Row Count Comparison Table

| Year | Expected | Datetime | Consumption | Weather | Merged | Missing Consumption | Missing Weather |
|------|----------|----------|-------------|---------|--------|-------------------|----------------|
| 2013 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2014 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2015 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2016 | 8785     | 8785     | 8761        | 8785    | 8761   | **-24**           | 0              |
| 2017 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2018 | 8761     | 8761     | 8750        | 8761    | 8750   | **-11**           | 0              |
| 2019 | 8761     | 8761     | 8748        | 8761    | 8748   | **-13**           | 0              |
| 2020 | 8785     | 8785     | 8742        | 8785    | 8742   | **-43**           | 0              |
| 2021 | 8761     | 8761     | 8735        | 8761    | 8735   | **-26**           | 0              |
| 2022 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2023 | 8761     | 8761     | 8761        | 8761    | 8761   | 0                 | 0              |
| 2024 | 8785     | 8785     | 8785        | 8785    | 8785   | 0                 | 0              |
| 2025 | 7297*    | 7202     | 7202        | 7248    | 7202   | (WIP)             | (WIP)          |

*2025 partial year (Jan 1 - Oct 29 = ~304 days = 7,296 hours expected + 1 header = 7,297) - Work in Progress

### Summary Statistics

- **Total Missing Consumption Hours**: 117 hours across 2016, 2018-2021 (excluding 2025 WIP)
- **Most Affected Year**: 2020 (43 missing hours from leap year)
- **Header Row Issue**: All files have +1 row (8761 instead of 8760 for regular years, 8785 instead of 8784 for leap years)
- **2025 Status**: Work in Progress - partial year data being developed
- **Merger Loss**: Additional 12 rows lost in final combination (112,289 → 112,277)

### Priority Issues

1. **HIGH**: Fix consumption data gaps for 2016, 2018-2021 (117 total missing hours)
2. **MEDIUM**: Standardize header counting across all processors (+1 row issue)
3. **MEDIUM**: Complete 2025 data collection (work in progress)
4. **LOW**: Fix final merger 12-row discrepancy
