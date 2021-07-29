# Data Loading and properties

## Loading Data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
plt.style.use('ggplot')

# To try using gpu
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
device = try_gpu()

# Load the data 
# If hourly write 1 if six hours write 6
data_rate = 6
days_solar_cycle = 27*2
# This is the shift in the indexes of the database to select the samples
index_shift = int(days_solar_cycle*24/data_rate)

if data_rate == 1:
    df = pd.read_csv('/content/gdrive/My Drive/LRI Stage/Processed_databases/Db_Sun_2010_2020_hourly.csv', index_col=0)
if data_rate == 6:
    df_train = np.load('Binary_train_54d_1999_2020_3_quarters.npy')
    df_test = np.load('Binary_test_54d_1999_2020_3_quarters.npy')
# Data taken from PLASMA: WIND HOURLY previously preprocessed
'''
# Visualizing the samples
x_plot = np.arange(0, days_solar_cycle, data_rate/24)
plot_int = np.random.randint(df_train.shape[0])
n_step_plots = 6
plt.figure(figsize=(10,6))
for plot, sample in enumerate(range(plot_int, plot_int+n_step_plots)):
    plt.step(x_plot, df_train[sample]+plot)
plt.xlabel('Days')
plt.ylabel("Samples")
plt.title("54-day Binarized Solar Wind Samples")
'''


# Convert dfs to torch format
X_train = torch.from_numpy(df_train).T.to(device).float()
X_test = torch.from_numpy(df_test).T.to(device).float()

# Load model
myRBM = torch.load('model_symm_sun_36_Symm_cells_216x360_20_epochs_lr_001_50_mcmc_steps_54_days.pt', map_location=try_gpu())
number_of_epochs = myRBM.total_epochs
myRBM.device = try_gpu()

# Half the learning rate (Comment to continue with the same lr)
# myRBM.learning_rate = myRBM.learning_rate/2. # Uncomment to half the learning rate

# Select the number of extra epochs:
extra_epochs = 20
number_of_epochs = myRBM.total_epochs + extra_epochs
myRBM.fit(X_train,X_test, epochs_max=extra_epochs)

# Save the progress in a new file
torch.save(myRBM, 'model_symm_sun_36_Symm_cells_216x360_40_epochs_lr_001_50_mcmc_steps_54_days.pt')

# Visualize properties of training
# Energy
plt.figure(figsize=(12,8))
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.energies_train, label='Train')
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.energies_test, label='Test')
plt.title('Mean energy per epoch')
plt.legend()
plt.xlabel('Epoch')
plt.show()

# Log-likelihood
plt.figure(figsize=(12,8))
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.log_likelihoods_train, label='Train')
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.log_likelihoods_test, label='Test')
plt.title('Mean Log-likelihood per epoch')
plt.legend()
plt.xlabel('Epoch')
plt.show()

# Singular values evolution
#@title Singular values full matrix { form-width: "300px" }
# Calculate the singular values of the full W matrix
singular_values_full_W = []
for matrix in myRBM.list_of_W_matrices[1:]:
    if myRBM.Symmetry_training:
        singular_values_full_W.append(torch.linalg.svd(myRBM._get_W_v_matrix(matrix))[1])
    if not myRBM.Symmetry_training:
        singular_values_full_W.append(torch.linalg.svd(matrix)[1])

plt.figure(figsize=(12,8))
n_sing_values_to_plot = 144
x_sing_vals = np.arange(len(singular_values_full_W))
if myRBM.Symmetry_training:
    x_sing_vals *= myRBM.Saving_interval
for svd_value in np.arange(n_sing_values_to_plot):
    plt.plot(x_sing_vals, [epoch[svd_value].item() for epoch in singular_values_full_W])
plt.title('First ' + str(n_sing_values_to_plot) + ' singular values of the full matrix per training epoch')
plt.xlabel('Training epoch')
plt.show()
