# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [X] look for columns that are easily and still obtainable via API endpoints for consultation
- [X] make clear boundaries of what is downloader, what is processor, what is final dataset assembler
- [X] add requirements for conda and pip, you can look generally for some environments and project settings overall
- [ ] gas net for consumption data is having maintanance so I need to process them later

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

### Notes after consultation

- firstly recreate the dataset as it is now, no new column needed
  - look into new source of **gasnet** data
- secondly try to find better source than that russian one for weather data, they are probably stealing it from somewhere else so I should be able to find it as well
- he said there is algorithm for calculating easter holiday, I should look into it
- for now, keep the before holiday and holiday as it is, we can resolve it later
- he was searching for some prazska plynarenska and chmi, might give it a look
