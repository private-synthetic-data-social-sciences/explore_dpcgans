
## Profiling the performance of DP CGAN

Documentation on how to profile and check the output.

Start with
```bash
poetry shell
```

Then
```bash
python -m cProfile -o profiling_output/nobs_XXXX_other_specs.txt src/train_sample_dpcgans.py --nobs 20000
```

where `profiling_X.txt` refers to a training run with specification `X` -- at the moment, this is just `10kobs`, ie, 10_000 observations. (see the script for details).

The output is stored in profiling_output. The files refer to the following specifications:
- nobs_10000.txt: train on 10k obs, sample (after training) on local laptop.
- nobs_10000_snellius.txt: train on 10k obs, sample (after training) on snellius. 
- nobs_10000_snellius_nosampling.txt: train on 10k obs, no sampling, on snellius. 
- nobs_20000_snellius_nosample_batchsize_2000.txt: train on 20k obs, no sampling, on snellius, with batch size 2000
    - tried the same as above with 30k obs, but cannot run it in a 15 minutes slot 


Then, run

```python
python -i src/inspect_profiling_results.py
```

or 
```bash
snakeviz profiling_results/filename.txt
```
