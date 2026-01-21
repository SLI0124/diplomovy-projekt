# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] notebook for checking missing values (for each year and in total; columns, rows, etc. - we might have some leakage that we need to find ASAP)
- [ ] EDA notebook (data have changed since last time, need to explore it again)
- [ ] check *float* and *int* values in final dataset, if one can be one or the other (for that little disk space saving and speed up)
- [ ] refactor notebook for machine learning part (right now it's a kinda mess that worked for initial experiments and presentation)

## Next steps

- [X] start date and end date should be set in one place (probably main script with help of `config.py` or `config.yml`) and passed to all the scripts and their functions, right now they are scattered across multiple scripts and it's hard to keep track of them
- [X] I've set default start date for all scripts to **2013.1.1**, right now I deleted param for start date, I think it won't be changed at all but it'd be nice to have a way to change it somehow, perhaps I can do like config file with dirs, start date, help functions, etc.
- [X] I have some duplicate code, create `utils.py` with all functions used across scripts that will work as some kind of config file (maybe even `config.py` or `config.yml` with dirs, start date, end date, etc. - need to think about what can be configured and what will be placed there)

---

- [ ] type checking, e.g. with add type hints to all functions and classes
- [ ] better docstrings with params, return types, etc., preferably to all functions and classes
- [ ] README file for `pipeline` directory, explain what each script does and how to use them, what are the inputs and outputs, conditions, etc.

---

- [ ] for processors try checking if `raw` files are even existing, if not, throw error or warning
  - DO NOT call downloader from processor and vice versa, implement single responsibility principle and don't mix those two things, do own implementation in each module
- [ ] speed up the processing the consumption data, right now I think it can be sped up somehow, it's kinda slow

---

- refactor
  - [ ] downloaders
  - [ ] processors
  - [ ] merging
  - [ ] utils
  - [ ] config
  - [ ] main script
- [ ] finally get into the modeling and AI part of the project

### Notes after consultation

- find some state-of-the-art models that are used for this kind of data and problem
- profit and foundation models, that was said to me, idk what exactly it means
- do some proper pipeline for random forest, xgboost etc.
- then I can use those models as baseline for more complex models
- fine-tune the params for those big models
- try to do multiple models for different time intervals
  - train till 2018 and test on 2019, etc. - incrementally widen the train set until the end of data where we have only few months to test on or something in similar fashion
- then we will see the breakpoint where model starts to fail, e.g. right after the Ukraine war started
