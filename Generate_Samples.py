# -*- coding: utf-8 -*-
"""Solar Wind GRBM.ipynb
Code created by Daniel Montoya
Original file is located at
    https://colab.research.google.com/drive/1y67dYmihDd2p8QZg7fMVd2cdm3G66NH3
"""



## Loading Data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import random
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
days_solar_cycle = 27*2 # In days
# This is the shift in the indexes of the database to select the samples
index_shift = int(days_solar_cycle*24/data_rate)
# Limit of analysis of generated data 
# (To avoid comparing at the end of the generated data, because the symmetry is strongly imposed)
limit_of_analysis = 27 # In days
limit_of_analysis_index = int(limit_of_analysis*24/data_rate)

if data_rate == 1:
    df = pd.read_csv('/content/gdrive/My Drive/LRI Stage/Processed_databases/Db_Sun_2010_2020_hourly.csv', index_col=0)
if data_rate == 6:
    df_train = np.load('Real_train_54d_1999_2020_3_quarters.npy')
    df_test = np.load('Real_test_54d_1999_2020_3_quarters.npy')
# Data taken from PLASMA: WIND HOURLY previously preprocessed
# Center the data
train_mean = df_train.mean()
train_data_std = df_train.std()
df_train -= train_mean
df_train /= train_data_std
df_test -= train_mean
df_test /= train_data_std

# Convert dfs to torch format
X_train = torch.from_numpy(df_train).T.to(device).float()
X_test = torch.from_numpy(df_test).T.to(device).float()

"""## Distributon properties

### Autocorrelation
"""

#@title 

# Defining our autocorrelation CHECKED allright
def my_Autocorrelation(visible, t):
    mean_0 = visible[:,0].mean()
    mean_t = visible[:,t].mean()
    return ( (visible[:,0] - mean_0) * (visible[:,t] - mean_t)).mean( axis=0 )/visible[:,0].std() / visible[:,t].std()

# Getting autocorrelations of data
data_autocorrelation_train = np.zeros((index_shift))
data_autocorrelation_test = np.zeros((index_shift))
for time in range(index_shift):
    data_autocorrelation_train[time] = my_Autocorrelation(df_train, time)
    data_autocorrelation_test[time] = my_Autocorrelation(df_test, time)


# Load RBM Pytorch trained model
# Native pytorch format
myRBM = torch.load('model_symm_36_cells_sun_GRBM_7000_epochs_lr_00008_50_mcmc_steps_lr_div2_epoch_1000_5000_3_4_data_std_descend.pt', map_location=try_gpu())
number_of_epochs = myRBM.total_epochs
myRBM.device = try_gpu()

"""#### Properties of training"""

plt.figure(figsize=(12,8))
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.energies_train, label='Train')
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.energies_test, label='Test')
plt.title('Mean energy per epoch')
plt.legend()
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.log_likelihoods_train, label='Train')
plt.plot(np.arange(0,number_of_epochs, myRBM.Saving_interval), myRBM.log_likelihoods_test, label='Test')
plt.title('Mean Log-likelihood per epoch G-Rbm ')
plt.legend()
plt.xlabel('Epoch')
plt.show()

#@title Singular values full matrix { form-width: "300px" }
# Calculate the singular values of the full W matrix
singular_values_full_W = []
for matrix in myRBM.list_of_W_matrices[1:]:
    if myRBM.Symmetry_training:
        singular_values_full_W.append(torch.linalg.svd(myRBM._get_W_v_matrix(matrix))[1])
    if not myRBM.Symmetry_training:
        singular_values_full_W.append(torch.linalg.svd(matrix)[1])

plt.figure(figsize=(12,8))
n_sing_values_to_plot = 100
x_sing_vals = np.arange(len(singular_values_full_W))
x_sing_vals *= myRBM.Saving_interval
for svd_value in np.arange(n_sing_values_to_plot):
    plt.plot(x_sing_vals, [epoch[svd_value].item() for epoch in singular_values_full_W])
plt.title('First ' + str(n_sing_values_to_plot) + ' singular values of the full matrix per training epoch')
plt.xlabel('Training epoch')
plt.show()

##################################################################################
# @title Singular values evolution { form-width: "200px" }
# How many plots to make:
rows_svd_plot = 2
columns_svd_plot = 4
first_cell_to_plot = 0

indexes_cells_plot = np.arange(first_cell_to_plot, myRBM.Symmetry_cells)
x_plot = np.arange(number_of_epochs/myRBM.Saving_interval)*myRBM.Saving_interval

fig, axs = plt.subplots(rows_svd_plot, columns_svd_plot, figsize=(5*columns_svd_plot,5*rows_svd_plot), sharey=True)
for row in range(rows_svd_plot):
    for column in range(columns_svd_plot):
        for singular_value in range(myRBM.v_units_per_cell):
            axs[row, column].plot(x_plot, [element[indexes_cells_plot[column+(row*columns_svd_plot)]][singular_value].item() for element in myRBM.list_of_singular_values_by_cell] )
        axs[row, column].set_title('Cell '+str(indexes_cells_plot[column+(row*columns_svd_plot)]))
        axs[row, column].set_xlabel('Epoch')
        axs[row, column].set_ylabel('Svd modes') 
plt.suptitle('Singular values evolution')
plt.tight_layout()
plt.show()
######################################################################################
# SVD Comparison
number_of_svd_lines = 5

svd_train = torch.linalg.svd(X_train)[1]
svd_test = torch.linalg.svd(X_test)[1]

plt.figure(figsize=(12,8))
for plot in np.linspace(0,myRBM.total_epochs/myRBM.Saving_interval-1, number_of_svd_lines, dtype=int):
    plt.plot(torch.linalg.svd(myRBM._get_W_v_matrix(myRBM.list_of_W_matrices[plot]))[1].cpu() , label='Epoch '+str(plot*myRBM.Saving_interval))
plt.plot(svd_train.cpu(), label='Train')
plt.plot(svd_test.cpu(), label='Test')
plt.legend()
plt.semilogy()
plt.semilogx()
plt.title('SVD comparison')
plt.xlabel('Wavenumber')
plt.ylabel('Value')
plt.show()

"""# Performance measurement

## Measures definition
"""

def MSE_gibbs_data(gibbs_samples, data):
    '''
    Returns the MSE of the second moment, meaning the mean squared error between the covariance matrix
    of data and gibbs generated data
    '''
    cov_mat_train = np.cov(data)
    cov_mat_gibbs = np.cov(gibbs_samples)
    return ((cov_mat_gibbs - cov_mat_train) * (cov_mat_gibbs - cov_mat_train)).mean()

# Create a function to generate samples based on gibbs sampling
def generate_gibbs_samples(model, steps, n_samples):
    # Generate random samples of the expected size
    new_samples_pytorch = torch.randint(low=0, high= 2, size=(model.Nv, n_samples), device=device).float()
    gibbs_samples_temp_pytorch_set_1 = new_samples_pytorch.clone()
    for gibbs_step in range(steps):
        # Gibbs sampling step
        hidden_temp_gibbs_pytorch, _ = myRBM.SamplerHiddens(gibbs_samples_temp_pytorch_set_1)
        gibbs_samples_temp_pytorch_set_2, _ = myRBM.SamplerVisibles(hidden_temp_gibbs_pytorch)
        gibbs_samples_temp_pytorch_set_1 = gibbs_samples_temp_pytorch_set_2.clone()
    return gibbs_samples_temp_pytorch_set_1.clone()

def Error_PSD(data1, data2):
    '''
    The data has to be in the format such that the data samples are the rows
    '''
    fourier_norm = 'ortho'
    spectrum1 = torch.fft.fft(data1.cpu(), dim=1, norm=fourier_norm).square().abs().mean(axis=0).log()
    spectrum2 = torch.fft.fft(data2.cpu(), dim=1, norm=fourier_norm).square().abs().mean(axis=0).log()
    return ((spectrum1-spectrum2)*(spectrum1-spectrum2)).sum().item()

"""## Generate samples"""
###############################################################################################
# Spacing of epochs to generate samples and examine
spacing_of_epochs = 2000
number_of_gibbs_sampling_steps = 1500
number_of_gibbs_samples = 5000
epochs_to_examine = np.arange(0,myRBM.total_epochs,spacing_of_epochs)

#@title Generate samples { form-width: "200px" }
list_of_means = []
list_of_MSE = []
list_of_likelihoods = []
list_of_PSD_errors = []
torch.manual_seed(52)
for epoch in tqdm(epochs_to_examine):
    if myRBM.Symmetry_training:
        myRBM._W_v_temp = myRBM._get_W_v_matrix(myRBM.list_of_W_matrices[int(epoch/myRBM.Saving_interval)])
        myRBM._W_h_temp = myRBM._get_W_h_matrix(myRBM.list_of_W_matrices[int(epoch/myRBM.Saving_interval)])
    if not myRBM.Symmetry_training:
        myRBM.W = myRBM.list_of_W_matrices[int(epoch/myRBM.Saving_interval)]
    myRBM.vbias = myRBM.list_of_vbias[int(epoch/myRBM.Saving_interval)].clone()
    myRBM.hbias = myRBM.list_of_hbias[int(epoch/myRBM.Saving_interval)].clone()
    gibbs_samples = generate_gibbs_samples(myRBM, number_of_gibbs_sampling_steps, number_of_gibbs_samples)
    list_of_means.append(gibbs_samples[:limit_of_analysis_index].mean())
    list_of_MSE.append(MSE_gibbs_data(gibbs_samples[:limit_of_analysis_index].cpu(), X_train[:limit_of_analysis_index].cpu()))
    list_of_likelihoods.append(myRBM.log_likelihood(gibbs_samples))
    list_of_PSD_errors.append(Error_PSD(X_train[:limit_of_analysis_index].t().cpu(), gibbs_samples[:limit_of_analysis_index].t().cpu()))

"""## Check behavior"""

# Build a dataframe from a dicionary of these values
data_dict = {'Means': [item.item() for item in list_of_means],
             'E2': [item.item() for item in list_of_MSE],
             'Likelihoods': [item.item() for item in list_of_likelihoods],
             'EPSD': list_of_PSD_errors}
df_data_gen = pd.DataFrame(data_dict)

# Save to csv for further analysis
df_data_gen.to_csv('GRbm_data_gen.csv', index=False)

plt.figure(figsize=(12,8))
plt.plot(epochs_to_examine, list_of_likelihoods, marker='v', label='Generated data LL S-Rbm')
#plt.plot(epochs_to_examine, no_sym_data['Likelihoods'], marker='o', label='Generated data LL Rbm')
plt.axhline(myRBM.log_likelihood(X_test), linestyle='--',color='b', label='Test data LL')
plt.axhline(myRBM.log_likelihood(X_train), linestyle='-.',color='b', label='Train data LL')
plt.title(r'Evolution of log-likelihoods of generated samples comparison')
plt.xlabel('Epoch')
plt.ylabel('Log-likelihood')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(epochs_to_examine, list_of_means, marker='v', label='Generated data mean S-Rbm')
#plt.plot(epochs_to_examine, no_sym_data['Means'], marker='o', label='Generated data mean Rbm')
plt.axhline(df_train.mean(), linestyle='--', label='Train data mean', color='b')
plt.axhline(df_test.mean(), linestyle='-.', label='Test data mean', color='b')
plt.title(r'Evolution of means of generated samples S-Rbm')
plt.legend()
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(epochs_to_examine, np.sqrt(list_of_MSE), marker='v', label=r'$\varepsilon^2$ S-Rbm')
#plt.plot(epochs_to_examine, np.sqrt(no_sym_data['E2']), marker='o', label=r'$\varepsilon^2$ Rbm')
plt.title(r'Evolution of $\varepsilon^2$ of generated samples S-Rbm')
plt.legend()
plt.ylabel(r'$\varepsilon^2$')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(epochs_to_examine[1:], np.sqrt(list_of_PSD_errors[1:]), marker='v', label=r'$\mathcal{E}^{PSD}$ S-Rbm')
#plt.plot(epochs_to_examine[1:], no_sym_data['EPSD'][1:], marker='o', label=r'$\mathcal{E}^{PSD}$ Rbm')
plt.title(r'Evolution of $\mathcal{E}^{PSD}$ of generated samples')
plt.xlabel('Epoch')
plt.ylabel(r'$\mathcal{E}^{PSD}$')
plt.legend()
plt.show()
#plt.ylim([0,110])


"""### Check properties of generated data"""

# Getting autocorrelations of gibbs samples
gibbs_autocorrelation_pytorch = torch.zeros((index_shift))
random_samples_autocorrelation = torch.zeros((index_shift))
for time in range(index_shift):
    gibbs_autocorrelation_pytorch[time] = my_Autocorrelation(gibbs_samples.t(), time)

# Plotting this autocorrelation with the test data
t_autocorrelation = np.arange(index_shift)*data_rate/24
plt.figure(figsize=(12,8))
plt.plot(t_autocorrelation, gibbs_autocorrelation_pytorch.detach(), label='Gibbs generated', marker='.')
plt.plot(t_autocorrelation, data_autocorrelation_test, label='Test data', marker='.')
plt.plot(t_autocorrelation, data_autocorrelation_train, marker='.', label='Train')
#plt.plot(t_autocorrelation, data_autocorrelation_synth_train, label='Synth data', marker='.')
#plt.plot(t_autocorrelation, random_samples_autocorrelation.detach(), label='Random data', marker='.')
plt.title('Autocorrelation of last gibbs sampling set Pytorch')
plt.xlabel('Days')
plt.legend()
plt.show()

"""# Looking at learned features"""
####################################################################################################
# How many plots to make:
rows_neuron_plot = 2
columns_neuron_plot = 4

# If using symmetry training
if myRBM.Symmetry_training:
    cell_number_to_examine = 0
    neurons_list = np.arange(myRBM.h_units_per_cell*cell_number_to_examine,myRBM.h_units_per_cell*(cell_number_to_examine+1))
    hidden_activation_one_neuron = torch.zeros((myRBM.h_units_per_cell,myRBM.Nh), device=myRBM.device, dtype=myRBM.dtype)
    for neuron_index,activation in zip(neurons_list,hidden_activation_one_neuron):
        activation[neuron_index] = 1

if not myRBM.Symmetry_training:
    neuron_number_to_examine = 0
    neurons_list = np.arange(neuron_number_to_examine,neuron_number_to_examine+(rows_neuron_plot*columns_neuron_plot))
    hidden_activation_one_neuron = torch.zeros((rows_neuron_plot*columns_neuron_plot,myRBM.Nh), device=myRBM.device, dtype=myRBM.dtype)
    for neuron_index,activation in zip(neurons_list,hidden_activation_one_neuron):
        activation[neuron_index] = 1


x_plot = np.arange(0, days_solar_cycle, data_rate/24)

#plt.figure(figsize=(15,15))
fig, axs = plt.subplots(rows_neuron_plot, columns_neuron_plot, figsize=(5*columns_neuron_plot,5*rows_neuron_plot))

for row in range(rows_neuron_plot):
    for plot,neuron in enumerate(hidden_activation_one_neuron[(row*columns_neuron_plot):(row+1)*columns_neuron_plot]):
        axs[row, plot].plot(x_plot, myRBM.SamplerVisibles(neuron.reshape(-1,1))[1].cpu() )
        axs[row, plot].set_title('Hidden neuron '+str(neurons_list[plot+(row*columns_neuron_plot)]))
        #axs[row, plot].set_ylim([0,1])
plt.suptitle('W features activations probaiblities')
plt.show()