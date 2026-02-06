# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] do the `ml` module first because a lot of those approaches do not require any kind of preprocessing, just raw data or simple scaling, so it will be quick to implement and test, also it will be good baseline for deep learning models when I will have more info about preprocessing, etc. from consultation (I can use preprocessed data later, raw data are ok now)
  - [ ] add more algorithms for **machine learning**, the more the merrier
- [X] update **machine learning** notebook with better structure and code organization, mostly more metrics
- [ ] read those two papers that he published and learn every key detail about the paper
- [X] maybe quickly add the Google model **TimesSF**, can be another baseline
- [ ] read those results thoroughly and try to draw some conclusions from them to be ready

## Next steps

- [X] rerun the notebook for deep learning models and see if scores are replicable
- [X] quickly go through dependencies and remove unused ones or add missing ones that are not specified in dependencies but are used in the code
- [ ] prepare for preprocessing consultation - do thorough research about extreme outliers and correlation to the real world events to check if those outliers are real or just data or measurement errors (this is important step, if I don't make it in time, propose that I will inform him via email somewhen later)
- [ ] maybe start writing? simple introduction should be easy and then you can think of mental model of how the rest of the paper will look like and you can have some ideas for consultation as well
- [ ] update project README with `uv` build tools and instructions
- [ ] chore: in `pipeline` module, move all arg endpoint at the end for consistency
- [ ] move **pipeline** to the `src` folder for better project structure, it makes more sense to have it there, update root path in `config.py` and run it once to make sure it works (tools can stay where they are, they are not really part of anything, damn now I am overthinking if moving pipeline is an option at all)
- when you may be really bored:
  - [ ] docs: some comments and docstrings are useless and obvious, remove the obvious ones or adjust them to be more informative, read through the code and add or remove comments as necessary
  - [ ] refactor: look at pipeline module

## Notes after consultation

- _to be continued.._

---

## Notes for consultation

- one-shot and parameter tuning? use both approaches?
- I think it is in paper but he uses different sliding window sizes for different models,  eg. 24 hours, 48 hours, 7 days, 14 days, 30 days, lag variables
- all other hyperparameter that can be used or considered for training those models
- maybe use some other dataset that is already battle-tested for time series forecasting with deep learning models?
- a lot of derivates of those state-of-the-art models exist, do I try small, medium, large, base or some other less know variants? do I focus more on well known but rather older models or try new rock stars?
  - [amazon/chronos-2](https://huggingface.co/amazon/chronos-2), [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base), [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small), [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large), [amazon/chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini), [autogluon/chronos-bolt-small](https://huggingface.co/autogluon/chronos-bolt-small), ...
  - is he familiar with some models or is it up to me?
- preprocessing is quite a tough nut to crack, ask as much as you can about it (I don't wanna do some stupid mistakes that will make whole project worthless so ask as much as you can and make as much notes and appointments as you can)  
  - do I need some fancy statistical methods? should I do some special transformations? ask everything that can be applicable, help here is welcomed
  - ask about how did he do preprocessing in his papers
  - ask about his code if possible from those papers
  - how to handle outliers properly? hopefully I can identify them on my own but how to treat them? remove, cap, transform, fill missing values?
  - tell him that in dissertation from Ing. Vojtěch Kotík he used dummy variables `is heating season` and `is summer`, by this I am referring to previous points about original source code
  - price is one day value replicated 24 times for each hour, is it ok? does it have some negative impact on training? is it better to have edge values and interpolate the rest? there it would help to have the source code
- ask him about what to ask to get some better GPU environment for training, I can't train and sleep at the same time or train and use my computer at any capacity
- tell him about initial thesis structure and chapters
