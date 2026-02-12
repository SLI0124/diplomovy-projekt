# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] `preprocessing` - a lot of things to get into, the earlier you start, the better
  - [ ] start with missing values and outliers
  - [ ] other complicated stuff just write down as you go for later research
- [ ] do the `ml` module first because a lot of those approaches do not require any kind of preprocessing, just raw data or simple scaling, so it will be quick to implement and test, also it will be good baseline for deep learning models when I will have more info about preprocessing, etc. from consultation (I can use preprocessed data later, raw data are ok now)
  - [ ] add more algorithms for **machine learning**, the more the merrier
- [ ] runnable `dl` scripts for foundation models
  - [ ] learn params can be used for training and use them as training parameters
  - [ ] for starters, set all models to one set of default params across all models
  - [ ] checkpoints and saving models

## Next steps

- [ ] plots, results, checkpoints, models, all you can think of to save during training and evaluation
- [ ] start writing - simple introduction should be easy and then you can think of mental model of how the rest of the paper will look like and you can have some ideas for consultation as well
- [ ] update project README with `uv` build tools and instructions
- [ ] chore: in `pipeline` module, move all arg endpoint at the end for consistency
- [ ] move **pipeline** to the `src` folder for better project structure, it makes more sense to have it there, update root path in `config.py` and run it once to make sure it works (tools can stay where they are, they are not really part of anything, damn now I am overthinking if moving pipeline is an option at all)
- when you may be really bored:
  - [ ] docs: some comments and docstrings are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] refactor: look at pipeline module

## Notes after consultation

- those three or four models are enough, no need to implement more
- make it modular via scripts
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
