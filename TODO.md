# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [X] check if legacy PPNet data are present for `--all` run, if not, throw warning or error, for no args and with args, it is nice to be aware of missing data or to be reminded to run the extractor first or request the data
- [X] ppnet cause a lot of missing values in final dataset, add flag for experimenting with and without ppnet data (e.g. `--no-ppnet`)
- [X] notebook for checking missing values (for each year and in total; columns, rows, etc. - we might have some leakage that we need to find ASAP)
  - did that internally, it is gonna be part of EDA notebook
- [X] check *float* and *int* values in final dataset, if one can be one or the other (for that little disk space saving and speed up)
  - I can't type **consumption_*** to be int, because of NaN values, so I think it's better to leave it as float, it rips me from inside
- [X] EDA notebook (data have changed since last time, need to explore it again)
- [x] quick shot of those state-of-the-art models via notebook and email it to supervisor that it works, ask him about closing the subject and next steps, model tuning, more preprocessing, more statistical hogwash, etc. (I still have two weeks before the deadline, so I can do quite a lot, only need to show proof of concept that those models can work with our data)
  - [x] find three distinct models that are used for time series forecasting
  - [x] one shot them without any tuning, just default params
  - [x] if time allows, try to do some basic hyperparameter tuning
  - [x] write down what you wanna say in the email to supervisor, very thoroughly
  - [x] document the results and send them to supervisor
- [ ] create `models` directory and move all model-related code there (**training**, **evaluation**, **preprocessing**, etc.), it will have`dl` and `ml` subdirs for deep learning and machine learning models respectively, maybe some **evaluation** dor or something in similar fashion
- [ ] update **machine learning** notebook with better structure and code organization

## Next steps

- [X] start date and end date should be set in one place (probably main script with help of `config.py` or `config.yml`) and passed to all the scripts and their functions, right now they are scattered across multiple scripts and it's hard to keep track of them
- [X] I've set default start date for all scripts to **2013.1.1**, right now I deleted param for start date, I think it won't be changed at all but it'd be nice to have a way to change it somehow, perhaps I can do like config file with dirs, start date, help functions, etc.
- [X] I have some duplicate code, create `utils.py` with all functions used across scripts that will work as some kind of config file (maybe even `config.py` or `config.yml` with dirs, start date, end date, etc. - need to think about what can be configured and what will be placed there)

---

- [X] type checking, e.g. with add type hints to all functions and classes
- [X] better docstrings with params, return types, etc., preferably to all functions and classes
- [X] README file for `pipeline` directory, explain what each script does and how to use them, what are the inputs and outputs, conditions, etc.

---

- refactor
  - [x] downloaders
  - [X] processors
  - [X] merging
  - [X] utils
  - [X] config
  - [x] main script

### Notes after consultation

- find some state-of-the-art models that are used for this kind of data and problem
- profit and foundation models, that was said to me, idk what exactly it means
- do some proper pipeline for random forest, xgboost etc.
- then I can use those models as baseline for more complex models
- fine-tune the params for those big models
- try to do multiple models for different time intervals
  - train till 2018 and test on 2019, etc. - incrementally widen the train set until the end of data where we have only few months to test on or something in similar fashion
- then we will see the breakpoint where model starts to fail, e.g. right after the Ukraine war started

## Notes for consultation

- one-shot and parameter tuning?
- a lot of derivates of those state-of-the-art models exist, do I try small, medium, large, base or some other less know variants?
- preprocessing is quite a tough nut to crack, ask as much as you can about it
  - I don't wanna do some stupid mistakes that will make whole project worthless so ask as much as you can and make as much notes and appointments as you can
