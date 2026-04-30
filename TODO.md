# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [X] for **EDA** do *SARIMAX*, probably to the notebook or create a new one, decide **ASAP**
- [X] delete code and logic for getting data from MLFlow and into tables and so on, we will redo it later, now it is mess
- [X] plots, results, checkpoints, models, plot and visualize all you can think of
  - [X] for starters do some basic EDA plots for the paper *(get into it ASAP to have more ideas for the paper, right now I don't have much of them)*
  - [X] plot fine-tuning and one-shot results, compare them, plot the delta between them (MLFlow has nice tools to get those data training and test results from db file), show some tables *(I had them but current code is a mess, I will redo it later)*
  - [X] I really like the prediction vs ground truth plots, they are very informative and can add insight into those breaking years we are interested in
- [X] **README.md** for *visualization*, what each script does, what are the requirement artifacts, etc.

## Next steps

- seasonal decomposition, trend and seasonality removal, fourier transforms, etc. are somewhat complicated but very common techniques for time series data preprocessing, if I have time later, I can look into it and add it to the pipeline, for now I will just write it down as a note for later research
  - nevermind, deep learning models are able to do all those things internally, those are for machine learning models such as random forest, xgboost, etc.

## Notes after consultation

- not anymore...

## Notes for consultation

- not anymore...
