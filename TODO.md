# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

### Deep Learning

- [ ] chronos2 might be too good and recent, it's not fair to compare it with older models (2024 vs 2025), look into chronos 1, which is named bolt and is released in 2024
- [ ] make 3 or 4 distinctive models for my own implementation, should look into basics like: LSTM, GRU, bi-directional LSTM, more or less hidden layers and count of neurons, sometimes convolution layers are used for time series data
  - [ ] those models will need many epochs to train, add it to the train script and log loss or some other metrics to MLFlow if that is possible, simple csv should be sufficient for that
- [ ] do script that will take all those training data from MLFlow db via library, maybe save it as json or keep it in memory and use it for plotting and comparing results of different models, all that is needed and necessary for the paper

### Others

- [ ] plots, results, checkpoints, models, plot and vizualize all you can think of
  - [ ] for starters do some basic EDA plots for the paper
  - [ ] plot fine-tuning and one-shot results, compare them, plot the delta between them (MLFlow has nice tools to get those data training and test results from db file), show some tables
  - [ ] I really like the prediction vs ground truth plots, they are very informative and can add insight into those breaking years we are interested in
- [ ] those very hard to understand and implement preprocessing steps are for machine learning, find some paper that mentions that those long domain knowledge for machine learning models are the same steps for deep learning models but they do it internally
- [ ] write
  - [ ] time series data and what are they
    - [ ] if you find that paper that shows that deep learning do those preprocessing steps internally, write about it here
    - [ ] crisis years and cross reference with some events/citations
  - [ ] machine learning models, briefly
  - [ ] deep learning models, more deeply (same chapter)
    - [ ] basics, layers, all that well known and widely used stuff
    - [ ] foundation SOTA models, their architecture, how they work, how they are trained, what are the differences between them, what are the differences between fine-tuning and one-shot
  - [ ] own models, metrics, training, results, experiments

## Next steps

- [ ] in all README files, format running scripts to the bash style, not PowerShell style, just use python and `name_of_script.py`, not `.\name_of_script.py`
- [ ] add to `tools` a **bash script** with all defaults, this script should download all data, preprocess it and train it and test it? there could be many of them, one for visualization, one for training, many options so this script will narrow it down and follow the path I used in the paper
- when you may be really bored:
  - [ ] docs: some comments and docstrings in `pipeline` are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] refactor: look at pipeline module
  - [ ] docs: write everywhere that every runnable script should be run from corresponding module, never from project root
  - [ ] style: I may be schizophrenic with code style, I should pick one and stick to it, look at ruff and linting rules, ruff should be achieving >99.9% Black compatibility
  - [ ] add `.vscode` folder, idk all I can put here but as I run this project on many machines atm it wouldn't hurt to have some common settings for vscode for formatting, linting, etc.
  - [ ] Dockerfile for easier setup and training, GPU support is
- [ ] notebooks overall redo, they should be more informative rather than just a playground, anything that run longer than 15 minutes should be moved to a script, should not be too hard at this point
- [ ] seasonal decomposition, trend and seasonality removal, fourier transforms, etc. are somewhat complicated but very common techniques for time series data preprocessing, if I have time later, I can look into it and add it to the pipeline, for now I will just write it down as a note for later research

## Notes after consultation

- *soon™*

## Notes for consultation

- I think it is in paper but he uses different sliding window sizes for different models,  eg. 24 hours, 48 hours, 7 days, 14 days, 30 days, lag variables **maybe later if I have time, include plots and results as appendix**
  - this can be further extended to different context window sizes AND at the same time lagged features
- all other hyperparameter that can be used or considered for training those models **maybe later if I have some time, include plots and results as appendix**
- a lot of derivates of those state-of-the-art models exist, do I try small, medium, large, base or some other less know variants? do I focus more on well known but rather older models or try new rock stars? **did not ask directly but since three big models are enough there is probably no need to include derivates**
  - [amazon/chronos-2](https://huggingface.co/amazon/chronos-2), [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base), [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small), [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large), [amazon/chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini), [autogluon/chronos-bolt-small](https://huggingface.co/autogluon/chronos-bolt-small), ...
  - is he familiar with some models or is it up to me? **it's all up to me**
- preprocessing is quite a tough nut to crack, ask as much as you can about it (I don't wanna do some stupid mistakes that will make whole project worthless so ask as much as you can and make as much notes and appointments as you can)  
  - do I need some fancy statistical methods? should I do some special transformations? ask everything that can be applicable, help here is welcomed **later if I have some time, include plots and results as appendix**
  - ask about how did he do preprocessing in his papers **don't be lazy and skim through those papers yourself**
- tell him about initial thesis structure and chapters **he proposed that, I better get on it**

---

- if I won't do any fancy preprocessing tied with seasonality and trend removal, do I still write about it in the paper?
- do I do some fancy preprocessing for my own models? get into training them then you will know for sure
