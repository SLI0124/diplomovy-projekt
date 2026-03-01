# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way. Get caffinated and let's get to work. You can **motherfucking do THIS!**

## Immediate

- [ ] `preprocessing` - a lot of things to get into so let's get into it *(to be fair have the most basic stuff done, get some results and ready them for consultation)*
  - [X] redo and check the logic, we wanna do it **once** and do it **right**, check everything, every missing or outlier value, check if it is real or data error, cross-reference with real world events, etc.
  - [ ] new scripts for preprocessing
    - [X] do filtering and filling as is described in [notebook](notebooks/0_preprocssing_eda.ipynb)
    - [ ] do research on **very basic** preprocessing methods for time series data to get started (*some ideas are below*)
      - [ ] remove date columns and replace it with single datetime column
      - [ ] add sin and cos transformations for cyclical features like hour of day, day of week, month of year, etc.
      - [ ] add lag target features, maybe some rolling mean or rolling std features, etc.

---

- [ ] runnable `dl` scripts for foundation models *(same here as for preprocessing, just get them working, very basic, then you can add )*
  - [ ] learn params can be used for training and use them as script parameters
    - [ ] basics tied to the any training process like learning rate, batch size, number of epochs, etc.
    - [ ] foundation time series models specific params like context window size, etc.
  - [ ] for starters, set all models to one set of default params across all models just to work, then set parameters unified e.g. context window size to 24 hours for all models, etc. (should not extend model capability that is not designed for it), maybe some config file for that or defaults
  - [ ] checkpoints and saving models, use `pytorch lightning` for easier training and saving if possible
  - [ ] logging, very important, preferably use some kind of tensorboard for pytorch, log everything you can think of, training and validation loss, metrics, hyperparameters, EVERYTHING (*Weights and Biases, TensorBoard, MLflow, Neptune/Comet/ClearML, PyTorch Lightning's built-in logger, etc.*)

---

- [ ] do the `ml` module, we need to compare them to the deep learning models to have comparable baseline results
  - [ ] add more algorithms for **machine learning**, the more the merrier (or some basic ones to cover baselines)

---

- [ ] plots, results, checkpoints, models, plot and vizualize all you can think of

---

- [X] other complicated preprocessing stuff just write down as you go for later research, he said to do basic stuff and then get to the complicated stuff if I have time
- [ ] read through new paper he sent me, overall do some reading, those papers and works from last year, prepare as much as possibly can before you write any code or text

## Next steps

- [ ] start writing - simple introduction should be easy and then you can think of mental model of how the rest of the paper will look like and you can have some ideas for consultation as well
- [X] update project README with `uv` build tools and instructions
- [X] update pipeline README with updated 403 error, that was cause with unauthorized access, I've added browser user agent to the request header and it seems to work
- [X] chore: in `pipeline` module, move all arg endpoint at the end for consistency
- [ ] move **pipeline** to the `src` folder for better project structure, it makes more sense to have it there, update root path in `config.py` and run it once to make sure it works (tools can stay where they are, they are not really part of anything, damn now I am overthinking if moving pipeline is an option at all)
- [ ] seasonal decomposition, trend and seasonality removal, fourier transforms, etc. are somewhat complicated but very common techniques for time series data preprocessing, if I have time later, I can look into it and add it to the pipeline, for now I will just write it down as a note for later research
- when you may be really bored:
  - [ ] docs: some comments and docstrings in `pipeline` are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] refactor: look at pipeline module
  - [ ] docs: write everywhere that every runnable script should be run from corresponding module, never from project root
  - [X] make pipeline modular, meaning everything besides main scripts should be importable and in module
  - [ ] style: I may be schizophrenic with code style, I should pick one and stick to it, look at ruff and linting rules, ruff should be achieving >99.9% Black compatibility
  - [X] chore: remove `ppnet`, at this point it is more of a liability than an asset
  - [x] build: configure `uv` and pytorch install process, make it compatible with GPU, CUDA and CPU
  - [ ] notebooks overall redo, they should be more informative rather than just a playground, anything that run longer than 15 minutes should be moved to a script, should not be too hard at this point
  - [ ] add `.vscode` folder, idk all I can put here but as I run this project on many machines atm it wouldn't hurt to have some common settings for vscode for formatting, linting, etc.
  - [ ] Dockerfile for easier setup and training, GPU support is

## Notes after consultation

- those three or four models are enough, no need to implement more
- make it modular via scripts, make checkpoints, save models, make it easy to run and test different approaches, this will be very helpful for writing the paper and recreating results later
- run those scripts, make some plots, tables, etc. just get some results for paper
- try to write about those in paper, this will get solid foundation for next steps
- preprocessing is key, no missing values because in time series it is a big no-no
  - fill missing values, he suggested machine learning models for that, regression models, maybe some basic mean or median imputation and look X hours before and after
  - he used zero to none preprocessing in his papers, he suggested very well known statistical methods like decomposition, trend and seasonality removal, fourier transforms, etc. (I read about it many times at this point, didn't know it'd be that basic and unimportant)
- use own architecture in later phase, he suggested it himself

## Notes for consultation

- one-shot and parameter tuning? use both approaches? **YES, do both**
- I think it is in paper but he uses different sliding window sizes for different models,  eg. 24 hours, 48 hours, 7 days, 14 days, 30 days, lag variables **maybe later if I have time, include plots and results as appendix**
- all other hyperparameter that can be used or considered for training those models **maybe later if I have some time, include plots and results as appendix**
- maybe use some other dataset that is already battle-tested for time series forecasting with deep learning models? **did not ask**
- a lot of derivates of those state-of-the-art models exist, do I try small, medium, large, base or some other less know variants? do I focus more on well known but rather older models or try new rock stars? **did not ask directly but since three big models are enough there is probably no need to include derivates**
  - [amazon/chronos-2](https://huggingface.co/amazon/chronos-2), [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base), [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small), [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large), [amazon/chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini), [autogluon/chronos-bolt-small](https://huggingface.co/autogluon/chronos-bolt-small), ...
  - is he familiar with some models or is it up to me? **it's all up to me**
- preprocessing is quite a tough nut to crack, ask as much as you can about it (I don't wanna do some stupid mistakes that will make whole project worthless so ask as much as you can and make as much notes and appointments as you can)  
  - do I need some fancy statistical methods? should I do some special transformations? ask everything that can be applicable, help here is welcomed **later if I have some time, include plots and results as appendix**
  - ask about how did he do preprocessing in his papers **don't be lazy and skim through those papers yourself**
  - ask about his code if possible from those papers **said that they did zero to none preprocessing, maybe something basic, this would be addition if I have time later**
  - how to handle outliers properly? hopefully I can identify them on my own but how to treat them? remove, cap, transform, fill missing values? **keep as much as you can, interpolate or use some regression models to fill missing values, cross-reference outliers with real world events to see if they are real or data errors**
  - tell him that in dissertation from Ing. Vojtěch Kotík he used dummy variables `is heating season` and `is summer`, by this I am referring to previous points about original source code **did not ask, probably not so important**
  - price is one day value replicated 24 times for each hour, is it ok? does it have some negative impact on training? is it better to have edge values and interpolate the rest? there it would help to have the source code
- ask him about what to ask to get some better GPU environment for training, I can't train and sleep at the same time or train and use my computer at any capacity **I did already send an email about it**
- tell him about initial thesis structure and chapters **he proposed that, I better get on it**
