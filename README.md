# Exploring DP-CGANS

The purpose of this repo is 
- to see how to use the generator in DP_CGANS and adapt it for usage with [`TAPAS`](https://github.com/alan-turing-institute/privacy-sdg-toolbox).
- to explore the performance of the DP_GANS package and find ways to speed it up.

## Contents 
The `src/` directory has code:
- `bgmm.ipynb`, `helpers.py`: notebook/helper functions exploring some performance aspect of the Bayesian Gaussian Mixture model in sklearn, that is used in DP CGAN
- `train_sample_dpcgans.py`: code to train and sample from DP CGAN.
- `inspect_profiling_results.py.py`: code to inspect profiling results.
- `sample_dpcgans.py.py`: deprecated code that opens a pickled model and uses the model to generate new data.

The `docs/` directory has markdown documents:
- `working-on-snellius.md`: how to run code on the supercomputer
- `performance-optimizaton.md`: overview of possibilities to improve the speed performance of DP CGAN.
- `profiling.md`: how to profile code and inspect the results.
- `proposal_svi.md`: Details for a proposal for improving the Gaussian Mixture Model with Stochastic Variational Inference.

## Setup

Requirements:
- python
- poetry 

```bash
git clone git@github.com:private-synthetic-data-social-sciences/explore_dpcgans.git
cd explore_dpcgans
poetry install 
poetry shell
# run stuff
```

