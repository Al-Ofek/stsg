# Sets file folder as working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

#Imports STSG Class
from stsg import STSG

# Ignore PreformanceWarning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

#For quantGAN
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import 
#%%
# Import data as dataframe
data = pd.read_csv('wave_height.csv', dtype=float)

data_name = 'wave_height'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# number of synthetic plots to show in the background of all graphs
n_plots = 100
dec_places = 4

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, data_start=0, data_end=2000, time_data=False)
#%% import data
path = f'{os.getcwd()}\\{data_name} generated data\\{data_name}'
synth_data_fourier = pd.read_csv(f'{path}_fourier.csv')
synth_data_ARMA = pd.read_csv(f'{path}_ARIMA.csv')
synth_data_GAN = pd.read_csv(f'{path}_GAN.csv')
synth_data_cosmos = pd.read_csv(f'{path}_cosmos.csv')

#%% OVERLAYED PLOT
rand = random.randint(0, 999)  # Pick out random synth series index

time_series_data = [model.data.iloc[:,0], synth_data_fourier.iloc[:,rand], synth_data_ARMA.iloc[:,rand],
                    synth_data_GAN.iloc[:,rand], synth_data_cosmos.iloc[:,rand]]
series_names = ['Original', 'Fourier', 'ARMA', 'GAN', 'CoSMoS']


# Create Subplots for Time Series, Distribution, and ACF
fig = plt.figure(figsize=(9, 6), dpi=300)

gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Plot parameters
lags = 10

colors = ['#245eaf','#377eb8','#4daf4a','#984ea3','#ff7f00']

# Overlay Time Series on the top plot
for i, data in enumerate(time_series_data):
    if i != 0:
        ax1.plot(data, label=series_names[i], color=colors[i], alpha=0.75)
    else:
        ax1.plot(data, label=series_names[i], linewidth=2, color=colors[i], zorder=10)
ax1.legend()

# Plot Distribution on the bottom left
for i, data in enumerate(time_series_data):
    if i==0:
        sns.histplot(data, bins=15, ax=ax2, kde=False,
                     label=series_names[i], element="step", stat="density",
                     color=colors[i], fill=True, alpha=0.35, zorder=0)
    else:
        sns.histplot(data, bins=15, ax=ax2, kde=False,
                 label=series_names[i], element="step", stat="density",
                 color=colors[i], fill=False, zorder = 5-i)
ax2.set_title('Distribution')
ax2.legend()

# Plot ACF on the bottom right
for i, data in enumerate(time_series_data):
    plot_acf(data, lags=lags, ax=ax3, color=colors[i], alpha=0.05, title=None)
ax3.set_title('ACF')
ax3.set_ylim(top=1.2, bottom=-1.2)

# Set labels
ax1.set_ylabel(model.data.name)
ax3.set_xlabel('Lags')
ax3.set_ylabel('Autocorrelation')

plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(export_dir, f'{data_name}_comparison_plot.png'), bbox_inches='tight')

#%% ORIGINAL
# Load Time Series Data
import random
rand = random.randint(0, 999) # Pick out random synth series index

time_series_data = [model.data.iloc[:,0], synth_data_fourier.iloc[:,rand], synth_data_ARMA.iloc[:,rand],
                    synth_data_GAN.iloc[:,rand], synth_data_cosmos.iloc[:,rand]]
series_names = ['Original', 'Fourier', 'ARMA', 'GAN', 'CoSMoS']

# Create Subplots for Time Series, Distribution, and ACF
fig, axes = plt.subplots(5, 3, figsize=(15, 15), gridspec_kw={'width_ratios': [2, 1, 1]},
                         dpi=300)

# Plot parameters
lags = 10

concat_data = np.concatenate(time_series_data)
min_val = np.min(concat_data)
max_val = np.max(concat_data)

bins = np.histogram_bin_edges(model.data.values, bins = 15, range=(min_val,max_val))

# Plot Time Series
for i, ax in enumerate(axes):
    # Time series plot
    axes[i, 0].plot(time_series_data[i])
    if i==2:
        axes[i, 0].set_title('ARMA')
    else:
        axes[i, 0].set_title(series_names[i])
    
    # Step 5: Plot Distribution
    sns.histplot(time_series_data[i], bins=bins, ax=axes[i, 1])
    axes[i, 1].set_xlabel(None)
    if i !=0:
        axes[i,1].set_ylabel(None)
    # axes[i, 0].set_title(series_names[i])
    # axes[i, 0].set_ylabel(series_names[i])

    # Step 6: Plot ACF
    plot_acf(time_series_data[i], lags=lags, ax=axes[i, 2], title=None)
    axes[i,2].set_ylim(top=1.2,bottom=-1.2)

axes[0, 0].set_ylabel(model.data.name)

axes[0, 1].set_title('Distribution')
axes[0,1].set_xlabel(model.data.name)

axes[0, 2].set_title('ACF')
axes[0,2].set_ylabel('Autocorrelation')
axes[0,2].set_xlabel('Lags')

plt.tight_layout()
plt.show()

fig.savefig(os.path.join(export_dir, f'{data_name} comparison plot'), bbox_inches='tight')