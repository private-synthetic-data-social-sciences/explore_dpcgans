"""Sample code to train and sample from a small DP-CGANS model"""

from dp_cgans import DP_CGAN
import pandas as pd
import torch 
import numpy as np 

np.random.seed(1)

assert torch.cuda.is_available(), "cuda not available"

tabular_data=pd.read_csv("../datasets/Adult/Real/real_adult_data.csv")

tabular_data = tabular_data.sample(n=20_000)

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

