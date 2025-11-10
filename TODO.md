# TODO

This will be a list of things that need to be done in order to finish the project. It's much easier to keep track of things this way.

## Immediate

- [ ] omg I found `ppnet` data source for consumption, I need to implement it instead of gasnet or perhaps add a parameter to choose which one to use, I could also download them all  
  - [ ] this would also mean I need to check main merging logic to avoid naming misconceptions
  - [ ] something with other datasets downloading them and merging them as well, I need to decipher what that email meant and what is required of me

## Next steps

- [ ] I've set default start date for all scripts to **2013.1.1**, right now I deleted param for start date, I think it won't be changed at all but it'd be nice to have a way to change it somehow, perhaps I can do like config file with dirs, start date, help functions, etc.
- [ ] I have some duplicate code, create `utils.py` with all functions used across scripts
- [ ] maybe check *float* and *int* values in final dataset, if one can be one or the other
- [ ] look for some other potential data source for new features (optional)
- [ ] type checking
- [ ] better docstrings with params, return types, etc.
- [ ] README file for `pipeline` directory
- [ ] for processors try checking if `raw` files are even existing, if not, throw error or warning, DO NOT call downloader from processor
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

### Notes for consultation

- empty as of now, need to look at that bloody email again

### Notes after consultation

- you can make new ones ;)
