# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

### Deep Learning

- [ ] support fine-tuning and one-shot learning, on top of that add uni-variate and covariant support (multivariate is mathematically and computationally more complex and it does not bring much to the table, a lot of those papers use covariant over multivariate approach)
  - [ ] make chronos2 and moirai1 support uni and covariant data
  - [ ] timesfm25 does it experimentally so I better check of that is possible
  - [ ] find another one or two models that support one-shot and fine-tuning, univariate and covariant data
- [ ] make 3 or 4 distinctive models for my own implementation, should look into basics like: LSTM, GRU, bi-directional LSTM, more or less hidden layers and count of neurons, sometimes convolution layers are used for time series data
  - [x] those models will need many epochs to train, add it to the train script and log loss or some other metrics to MLFlow if that is possible, simple csv should be sufficient for that
  - [x] add for testing true and predicted values
  - [ ] also support uni and covariant data
- [ ] do script that will take all those training data from MLFlow db via library, maybe save it as json or keep it in memory and use it for plotting and comparing results of different models, all that is needed and necessary for the paper

### Others

- [ ] plots, results, checkpoints, models, plot and vizualize all you can think of
  - [ ] for starters do some basic EDA plots for the paper
  - [ ] plot fine-tuning and one-shot results, compare them, plot the delta between them (MLFlow has nice tools to get those data training and test results from db file), show some tables
  - [ ] I really like the prediction vs ground truth plots, they are very informative and can add insight into those breaking years we are interested in
- [ ] write
  - [ ] time series data and what are they
    - [ ] if you find that paper that shows that deep learning do those preprocessing steps internally, write about it here
  - [ ] machine learning models, briefly
  - [ ] deep learning models, more deeply (same chapter)
    - [ ] basics, layers, all that well known and widely used stuff
    - [ ] foundation SOTA models, their architecture, how they work, how they are trained, what are the differences between them, what are the differences between fine-tuning and one-shot
  - [ ] collecting my own data, preprocessing it, metrics, extreme outliers, basic EDA
    - [ ] crisis years and cross reference with some events/citations
  - [ ] own models, metrics, training, results, experiments

## Next steps

- [ ] in all README files, format running scripts to the bash style, not PowerShell style, just use python and `name_of_script.py`, not `.\name_of_script.py`
- [ ] add to `tools` a **bash script** and **PowerShell script** with all defaults, this script should download all data, preprocess it and train it and test it? there could be many of them, one for visualization, one for training, many options so this script will narrow it down and follow the path I used in the paper or give decent look into workflow
- when you may be really bored:
  - [ ] docs: some comments and docstrings in `pipeline` are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] refactor: look at pipeline module
  - [ ] docs: write everywhere that every runnable script should be run from corresponding module, never from project root
  - [X] add `.vscode` folder, idk all I can put here but as I run this project on many machines atm it wouldn't hurt to have some common settings for vscode for formatting, linting, etc.
- [ ] notebooks overall redo, they should be more informative rather than just a playground, anything that run longer than 15 minutes should be moved to a script, should not be too hard at this point

---

- seasonal decomposition, trend and seasonality removal, fourier transforms, etc. are somewhat complicated but very common techniques for time series data preprocessing, if I have time later, I can look into it and add it to the pipeline, for now I will just write it down as a note for later research
  - nevermind, deep learning models are able to do all those things internally, those are for machine learning models such as random forest, xgboost, etc.

## Notes after consultation

- *soon™*

## Notes for consultation

- *soon™*
