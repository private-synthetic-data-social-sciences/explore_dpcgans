

# Performance optimization for DP-CGANs

This is all using DP CGAN with `private=False`.

## Findings

### Profiling output

See the snakeviz output:
```bash
snakeviz profiling_output/nobs_20000_snellius_nosample_batchsize_2000.txt
```


- `data_transformer.py` because it is taking a lot of time overall (almost 50%); some things are also not using their time efficiently (`_estimate_log_gaussian_prob`)
    - It seems like it has bad time complexity also for the number of rows, but I should check this more thoroughly
    - Specifically, it scales in O(f(n)) * O(m), where m is the number of columns and n is the number of rows: the O(m) comes from the for loop across columns; f(n) is the complexity of a single run of `sklearn.mixture.bgmm.fit()`, which I don't know what it is exactly
- `sample_condvec_pair` (which has some inefficient use of time)
- `.backward` is still a little inefficient, taking 6% of the total and not splitting well into children
    - perhaps check this with different batch sizes
- by tottime
    - reduce of numpy ufunc
    - `_gaussian_mixture.py`, in particular `_gaussian_mixture.py` uses the time very inefficiently



### Checking the source code 

There is a nested for loop in `data_transformer.py` -- this matters for primarily when adding columns, secondarily when adding rows (rows only when they have new values)
- https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L91: iterates over the columns, calls `_fit_discrete()` or `_fit_continuous()`
- https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L114: iterates over the unique (continuous?) values in the column?
- it seems like it's taken from [`ctgan/data_transformer.py`](https://github.com/sdv-dev/CTGAN/blob/0de42b497eb207ba96ea8aa6f101888367de1347/ctgan/data_transformer.py#L4); the implementation there is very similar
    - but `_fit_continuous` is modified for DP-CGAN. https://github.com/sunchang0124/dp_cgans/blob/0ed8b5c0f1307decb2d76ede76edfbe39fc195db/src/dp_cgans/data_transformer.py#L34
- `_fit_continuous` calls [`sklearn.mixture.BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) for each column separately. 

### Other 
There is a memory leak issue with the scipy version used: https://github.com/private-synthetic-data-social-sciences/explore_dpcgans/security/dependabot/2
Not sure it's a big issue here; would need to update the scipy dependency in DP-CGAN. 

## Ideas for improvements

### Parallelize
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
- But the BGMM routine already utilizes all CPU cores, so I don't think there is much to gain from parallelizing this 

### Streamline the data transformation
- Is there room for optimizations within a single call to `_fit_continuous` or `_fit_discrete`. For this, I need to profile one call to these functions.


### Improve the BGMM routine 

- Change parameters
    - The `init=5` flag I think is the biggest bottleneck
        - In RDT, they use `n_init=1`, and it seems to scale linearly in `n_init` in my small experiments
        - What was the motivation to use `n_init>1`?
        - TODO: when does `n_init` matter when `warm_start=True`? The documentation says `n_init` is ignored when `warm_start=True`, but that perhaps only holds when the object is reused? Ie, at the first initialization, `warm_start` does not matter (there's nothing to start from), and `n_init` is binding.
    - Initializing with "kmeans" instead of random may also give some speedup 
    - I am less sure about the specifications for the covariance type 
- Re-use BGMM model across runs 
    - this is a speed-up only for the privacy assessment 
    - we should first double-check that this indeed does what we expect it to
    - for instance, pre-train the BGMM model on a 50% split of the input data, and do the privacy assessment only on the remaining 50%
    - this will be a bigger programming effort I think
- Use GPU 
    - unclear whether existing libraries can be used 
- Other questions
    - [Why is the number of components hard-coded?] It seems more flexible to set the number of components higher, and let the model decide if they are not needed? (see documentation)
        - the running time seems to increase quickly with the number of components
        - the time complexity also seems to depend multiplicatively on the number of components and on the size of the input (see notebook).
        - in RDT, they use a default of 10

## Conclusion

We should explore the options in the following order 
1. Vary the batch size, if not done already
    - What are realistic datasets? 
    - It depends on each dataset. 
2. The implementation of the `data_transfomer`:
    1. Get rid of the `n_init=5` flag. 
    3. Re-use the BGMM model across runs of DP-CGAN -- but that only helps if the estimation does a lot of iterations. Need to find this out.
    4. Explore options to use Stochastic VI (with or without GPU):
        - https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf
        - https://arxiv.org/abs/1601.00670
    4. use the GPU for estimating the BGMM. 
