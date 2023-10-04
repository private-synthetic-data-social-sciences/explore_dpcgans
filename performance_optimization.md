

## Performance optimization for DP-CGANs

### Main observations
- The profiler results show that `data_transformer.py:35(_fit_continuous)` takes more than 1/3 of the total execution time of training DP CGAN once (see profiler output below)
- `dp_cgans.data_transformer._fit_discrete` calls `pd.DataFrame` once, while the original implementation does not. https://github.com/sdv-dev/CTGAN/blob/0de42b497eb207ba96ea8aa6f101888367de1347/ctgan/data_transformer.py#L4. is this important for performance?
    - for `_fit_continuous`, Chang uses a `sklearn.mixture.BayesianGaussianMixture()` instead of `rdt.transformers.numerical.ClusterBasedNormalizer`. why? what is the time complexity here?



```python
    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    2884/1    0.035    0.000  540.053  540.053 {built-in method builtins.exec}
        1    0.001    0.001  540.053  540.053 train_sample_dpcgans.py:1(<module>)
        1    0.000    0.000  538.625  538.625 base.py:117(fit)
        1    0.002    0.002  538.598  538.598 dp_cgan_init.py:25(_fit)
        1    4.563    4.563  538.594  538.594 dp_cgan.py:356(fit)
        1    0.000    0.000  161.155  161.155 data_transformer.py:73(fit)
        7    0.000    0.000  161.140   23.020 data_transformer.py:35(_fit_continuous)
        7    0.000    0.000  161.140   23.020 _base.py:171(fit)
        7    0.193    0.028  161.140   23.020 _base.py:196(fit_predict)
    20000  133.583    0.007  133.583    0.007 {method 'run_backward' of 'torch._C._EngineBase' objects}
    15000    0.106    0.000  132.990    0.009 _tensor.py:429(backward)
    15000    0.167    0.000  132.876    0.009 __init__.py:103(backward)
    10000   24.863    0.002  106.647    0.011 data_sampler.py:241(sample_condvec_pair)
13354225/13238546   21.000    0.000  103.692    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
    28608    0.073    0.000   90.144    0.003 _base.py:282(_e_step)
    28615    4.593    0.000   89.089    0.003 _base.py:484(_estimate_log_prob_resp)
    28601    6.347    0.000   58.506    0.002 _bayesian_mixture.py:664(_m_step)
6559179   55.388    0.000   55.388    0.000 {method 'reduce' of 'numpy.ufunc' objects}
3572314    4.948    0.000   49.217    0.000 fromnumeric.py:69(_wrapreduction)
    28615   13.151    0.000   45.261    0.002 _logsumexp.py:7(logsumexp)
```


#### Checking the source code 

There is a nested for loop in `data_transformer.py` -- this matters for primarily when adding columns, secondarily when adding rows (rows only when they have new values)
- https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L91: iterates over the columns, calls `_fit_discrete()` or `_fit_continuous()`
- https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L114: iterates over the unique (continuous?) values in the column?
- it seems like it's taken from [`ctgan/data_transformer.py`](https://github.com/sdv-dev/CTGAN/blob/0de42b497eb207ba96ea8aa6f101888367de1347/ctgan/data_transformer.py#L4); the implementation there is very similar
    - but `_fit_continuous` is modified for DP-CGAN. https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L34
- `_fit_continuous` calls [`sklearn.mixture.BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) for each column separately. 


### Improvement: parallelize
- The Bayesian Mixture models for each column are independent of any other column; therefore, this is conveniently parallel.
- (Note that efficient parallelization would parallelize across *all* columns, which can be discrete or continuous and which requires different function)
- What are the options?
    - JIT/numba: out of the question because it's not numpy directly
    - Thread-based or process-based parallelism / joblib / dask?
        - This would be most straightforward to implement.
        - https://ml.dask.org/joblib.html
    - One could think of using the GPU for this
        - Since the data will eventually need to be on the GPU for training the model
        - But this seems more painful to implement.
        - [`pycave`](https://pycave.borchero.com/index.html) has Gaussian Mixtures, but not Bayesian Gaussian Mixture. I'm not sure how active this package is.
        - Google says there are some new GPU implementations of Bayesian Gaussian Mixture Models, for instance [this paper](https://www.jstage.jst.go.jp/article/transinf/E105.D/3/E105.D_2021EDP7121/_pdf)

### Improvement: streamline transformation
- Is there room for optimizations within a single call to `_fit_continuous` or `_fit_discrete`. For this, I need to profile one call to these functions.

