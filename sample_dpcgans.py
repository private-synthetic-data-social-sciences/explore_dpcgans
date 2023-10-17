import pickle

import torch

assert torch.cuda.is_available(), "cuda not available"

filename = "/home/fhafner/gans/datasets/Adult/DPCGANS/dpcgans_train_2000.pkl"
filename = "/gpfs/home2/hafner/projects/gans/datasets/Adult/DPCGANS/dpcgans_train_2000.pkl"
# filename = "/gpfs/home2/hafner/projects/gans/datasets/Census/DPCGANS/dpcgans_train_2000.pkl"

# load the model
with open(filename, "rb") as f:
    model = pickle.load(f)


# try to sample
model.sample()

# model.sample(n=2, condition_column="age", condition_value="30")
# model.sample(2) # TypeError: 'NoneType' object is not iterable
# model.sample(num_rows=3) # TypeError: 'NoneType' object is not iterable
