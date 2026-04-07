# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] add default date to the pipeline as 2025-12-31 to replicate state of thesis data
- [ ] plot image *apparent_temperature_all_years_trend.png* is not used, delete and script
- [X] do script that will take all those training data from MLFlow db via library, maybe save it as json or keep it in memory and use it for plotting and comparing results of different models, all that is needed and necessary for the paper, do like one file that is responsible for getting those data, then each file corresponding to role it should fulfill, one for plotting, one for comparing results, making plots, etc.
- [x] do simpler custom model with straightforward data preprocessing, no branching, we are aiming below 10 MAPE/SMAPE and 10 epochs
- [ ] delete all **machine learning** related code, remove `ml` module since I am not using it anymore
- [ ] for **EDA** do *SARIMAX*, probably to the notebook or create a new one, decide **ASAP**
- [ ] plots, results, checkpoints, models, plot and visualize all you can think of
  - [ ] for starters do some basic EDA plots for the paper
  - [ ] plot fine-tuning and one-shot results, compare them, plot the delta between them (MLFlow has nice tools to get those data training and test results from db file), show some tables
  - [x] I really like the prediction vs ground truth plots, they are very informative and can add insight into those breaking years we are interested in
- [ ] **README.md** for *visualization*, what each script does, what are the requirement artifacts, etc.
- [ ] delete all other branches
- [ ] do branch for final hand in

## Next steps

- [ ] add to `tools` a **bash script** and **PowerShell script** with all defaults, this script should download all data, preprocess it and train it and test it? there could be many of them, one for visualization, one for training, many options so this script will narrow it down and follow the path I used in the paper or give decent look into workflow
  - [ ] download data
  - [ ] preprocess data
  - [ ] training models
    - [ ] foundation models
    - [ ] own models
  - [ ] testing models
    - [ ] foundation models
      - [ ] fine-tuning
      - [ ] one-shot
    - [ ] own models
      - [ ] fine-tuning
- when you may be really bored:
  - [ ] docs: some comments and docstrings in `pipeline` are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] docs: write everywhere that every runnable script should be run from corresponding module, never from project root
- [ ] notebooks overall redo, they should be more informative rather than just a playground, anything that run longer than 15 minutes should be moved to a script, should not be too hard at this point
  - [ ] remove `ml` notebook
  - [ ] replace `ml` notebook with `sarimax` (or add to the EDA notebook, decide later, not a big deal)
  - [ ] for `dl` take model, test it and print some stats, just for the fun that it works (we wanna take granite probably because it is smallest size and it will be included in source code)

---

- seasonal decomposition, trend and seasonality removal, fourier transforms, etc. are somewhat complicated but very common techniques for time series data preprocessing, if I have time later, I can look into it and add it to the pipeline, for now I will just write it down as a note for later research
  - nevermind, deep learning models are able to do all those things internally, those are for machine learning models such as random forest, xgboost, etc.

## Notes after consultation

- *soon™*

## Notes for consultation

- *soon™*
