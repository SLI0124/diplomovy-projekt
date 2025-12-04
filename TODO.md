# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] I found `ppnet` data source for consumption, right now I have multiple sources for consumption so it wouldn't be such a big hassle to add it but their most oldest data I can download is from **2015.1.1**
  - one way is to completely ignore it and keep current sources
  - another way is to upload already existing data I have from previous projects that was given to me

## Next steps

- [ ] I've set default start date for all scripts to **2013.1.1**, right now I deleted param for start date, I think it won't be changed at all but it'd be nice to have a way to change it somehow, perhaps I can do like config file with dirs, start date, help functions, etc.
- [ ] I have some duplicate code, create `utils.py` with all functions used across scripts
- [ ] maybe check *float* and *int* values in final dataset, if one can be one or the other
- [ ] type checking
- [ ] better docstrings with params, return types, etc.
- [ ] README file for `pipeline` directory
- [ ] for processors try checking if `raw` files are even existing, if not, throw error or warning
  - DO NOT call downloader from processor and vice versa, implement single responsibility principle and don't mix those two things, do own implementation in each module
- [ ] set up some black, pylint or other code formatting and linting tools for repository project
- more notebooks:
  - [ ] check merging of each year of each column and in total
  - [ ] data analysis of missing values
- [ ] tests for merging?
- refactor
  - [ ] downloaders
  - [ ] processors
  - [ ] merging
  - [ ] utils
  - [ ] config
  - [ ] main script
  - [ ] jupyter notebooks

### Notes after consultation

- find some state-of-the-art models that are used for this kind of data and problem
- profit and foundation models, that was said to me, idk what exactly it means
- do some proper pipeline for random forest, xgboost etc.
- then I can use those models as baseline for more complex models
- fine-tune the params for those big models
- try to do multiple models for different time intervals
  - like train till 2018 and test on 2019, etc.
- then we will see the breakpoint where model starts to fail, e.g. right after the Ukraine war started
