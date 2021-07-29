# Data Loading and properties

## Loading Data

# This is to be able to load the scripts from the files

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
# Device can also be GPU
# device = torch.device("cpu")
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

# Training the RBM

# Define number of hidden variables (units)
Symmetry_cells = 18*2
number_of_hidden_units = 10*Symmetry_cells
number_of_epochs = 30

# Convert dfs to torch format
X_train = torch.from_numpy(df_train).T.to(device).float()
X_test = torch.from_numpy(df_test).T.to(device).float()

### Using Pytorch code


# Load RBM Pytorch implementation
from rbm_symm import RBM
# from RBM_pytorch.rbm_1 import RBM

myRBM = RBM(num_visible = df_train.shape[1],
            num_hidden = number_of_hidden_units,
            device = device,
            mini_batch_size=100,
            learning_rate = 0.01,
            gibbs_steps=50,
            Symmetry_training=True,
            Symmetry_cells=Symmetry_cells,
            Saving_interval = 10) 

# Train the model
torch.manual_seed(30)
myRBM.fit(X_train, X_test, epochs_max=number_of_epochs)

torch.save(myRBM, 'model_symm_sun_36_Symm_cells_216x360_20_epochs_lr_001_50_mcmc_steps_54_days.pt')
print('Done!')

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
x_sing_vals *= myRBM.Saving_interval
for svd_value in np.arange(n_sing_values_to_plot):
    plt.plot(x_sing_vals, [epoch[svd_value].item() for epoch in singular_values_full_W])
plt.title('First ' + str(n_sing_values_to_plot) + ' singular values of the full matrix per training epoch')
plt.xlabel('Training epoch')
plt.show()
