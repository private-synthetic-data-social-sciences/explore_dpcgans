"""Sample code to train and sample from a small DP-CGANS model"""

import argparse

import numpy as np
import pandas as pd
import torch
from dp_cgans import DP_CGAN
import socket 

parser = argparse.ArgumentParser()
parser.add_argument("--nobs", 
                    default = 10_000, 
                    type=int,
                    help="Number of observations to draw from the original data")

args = parser.parse_args()
np.random.seed(1)

hostnames_without_gpu = ["flavio-XPS-13-9310"]
hostname = socket.gethostname()


if hostname not in hostnames_without_gpu:
   assert torch.cuda.is_available(), "cuda not available"

tabular_data=pd.read_csv("../datasets/Adult/Real/real_adult_data.csv")

if args.nobs < tabular_data.shape[0]:
   tabular_data = tabular_data.sample(n=args.nobs)

# We adjusted the original CTGAN model from SDV. Instead of looking at the distribution of individual variable, we extended to two variables and keep their corrll
model = DP_CGAN(
   epochs=100, # number of training epochs
   batch_size=2000, # the size of each batch
   log_frequency=True,
   verbose=True,
   generator_dim=(128, 128, 128),
   discriminator_dim=(128, 128, 128),
   generator_lr=2e-4, 
   discriminator_lr=2e-4,
   discriminator_steps=1, 
   private=False,
)

print("Start training model")
model.fit(tabular_data)
model.save("generator.pkl")

# Generate 100 synthetic rows
# syn_data = model.sample(100)
# syn_data.to_csv("syn_data_file.csv")
