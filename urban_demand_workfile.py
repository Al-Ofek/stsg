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

#%% Model Creation

# Import data as dataframe
data = pd.read_csv('urban_demand.csv')

data_name = 'urban_demand'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Sets the data start/end times
start = '01/07/2020 00:00'
end = '01/08/2020 00:00'

# number of synthetic plots to show in the background of all graphs
n_plots = 100

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, data_start=start, data_end=end)

#%% Fourier multiple plots (diff. m)
model.plot_fourier(m_values=[3,25,80], n_samples=100)
plt.savefig(os.path.join(export_dir, f'{data_name} fourier w diff coeff.'), bbox_inches='tight')

#%% Fourier
# Generate data using Fourier
synth_data_fourier = model.gen_fourier(keep_phases=3, n_samples=1000, non_negative=True)
#%%
# save to file
# synth_data_fourier.to_csv(os.path.join(export_dir, f'{data_name}_fourier.csv'))

# Plot original + synthesized data - 5 instances only
model.plot(s_data=synth_data_fourier.iloc[:,:n_plots],
                                                filename=f'{data_name}_fourier')
# similarity scores
fourier_dtw = model.dtw_score(synth_data_fourier)
fourier_wass = model.wass_dist(synth_data_fourier)

fourier_dtw.to_csv(os.path.join(export_dir, f'{data_name}_fourier_dtw.csv'))
fourier_wass.to_csv(os.path.join(export_dir, f'{data_name}_fourier_wass.csv'))

# statistics
fourier_stats = model.calculate_statistics(synth_data_fourier)
# fourier_stats.to_csv(os.path.join(export_dir, f'{data_name}_fourier_stats.csv'))


#%% ARIMA
# Generate data using ARIMA
synth_data_ARIMA = model.gen_ARIMA(p=3, d=0, q=3, n_samples=1000)

#save to file
synth_data_ARIMA.to_csv(os.path.join(export_dir, f'{data_name}_ARIMA.csv'))

# Plot original + synthesized data
model.plot(s_data=synth_data_ARIMA.iloc[:,:n_plots],
                                           filename=f'{data_name}_ARIMA')

# similarity scores
ARIMA_dtw = model.dtw_score(synth_data_ARIMA)
ARIMA_wass = model.wass_dist(synth_data_ARIMA)

ARIMA_dtw.to_csv(os.path.join(export_dir, f'{data_name}_ARIMA_dtw.csv'))
ARIMA_wass.to_csv(os.path.join(export_dir, f'{data_name}_ARIMA_wass.csv'))

#statistics
ARIMA_stats = model.calculate_statistics(synth_data_ARIMA)
ARIMA_stats.to_csv(os.path.join(export_dir, f'{data_name}_ARIMA_stats.csv'))

#%% GAN loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

# Dataset
class og_Dataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = np.expand_dims(self.data[index:index+self.window], -1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.data) - self.window

# GAN model
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding='same')

        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same')
        self.relu2 = nn.PReLU()

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=80):
        super(TCN, self).__init__()
        layers = []
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            dilation = 2 * dilation if i > 1 else 1
            layers += [TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation)]
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.net(x.transpose(1, 2))
        return self.conv(y1).transpose(1, 2)

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.tanh(self.net(x))

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.net(x))
    
GAN_path = f'{os.getcwd()}\\GAN\\{data_name} Generator\\{data_name}_epochs=500_batch=6_seq=127'
GAN_gen = torch.load(GAN_path, map_location=torch.device('cpu'))

#%% GAN Generator

series_length = len(model.data)
n_samples = 1000
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform(model.data.array.reshape(-1,1))[:,0]

d = {}
for k in range(n_samples):
    noise = torch.randn(1, series_length, 3, device=device)
    synth_series = GAN_gen(noise).detach().cpu().reshape(series_length).numpy()
    synth_series = np.reshape(synth_series, (-1,1))
    synth_series_adj = scaler.inverse_transform(synth_series)
    d[model.data.name+'_'+str(k+1)] = synth_series_adj[:,0]

synth_data_GAN = pd.DataFrame(d)
#%% GAN Analysis
# Plot original + synthesized data
model.plot(s_data=synth_data_GAN.iloc[:,:n_plots],
                                                filename=f'{data_name}_GAN')
# similarity scores

GAN_dtw = model.dtw_score(synth_data_GAN)
GAN_wass = model.wass_dist(synth_data_GAN)

GAN_dtw.to_csv(os.path.join(export_dir, f'{data_name}_GAN_dtw.csv'))
GAN_wass.to_csv(os.path.join(export_dir, f'{data_name}_GAN_wass.csv'))

# statistics
GAN_stats = model.calculate_statistics(synth_data_GAN)
GAN_stats.to_csv(os.path.join(export_dir, f'{data_name}_GAN_stats.csv'))

#%% CoSMoS
# Import from R
df = pd.read_csv(f'{export_dir}\\urban_demand_cosmos.csv')
n_samples = len(df.columns)
d = {}
for k in range(n_samples):
    d[model.data.name+'_'+str(k+1)] = df.iloc[:,k]

synth_data_cosmos = pd.DataFrame(d)

#%%
# Plot original + synthesized data
model.plot(s_data=synth_data_cosmos.iloc[:,:n_plots],
                                                filename=f'{data_name}_cosmos')
# similarity scores
cosmos_dtw = model.dtw_score(synth_data_cosmos)
cosmos_wass = model.wass_dist(synth_data_cosmos)

cosmos_dtw.to_csv(os.path.join(export_dir, f'{data_name}_cosmos_dtw.csv'))
cosmos_wass.to_csv(os.path.join(export_dir, f'{data_name}_cosmos_wass.csv'))

# statistics
cosmos_stats = model.calculate_statistics(synth_data_cosmos)
cosmos_stats.to_csv(os.path.join(export_dir, f'{data_name}_cosmos_stats.csv'))

#%% Summary
rows = ['Original', 'Fourier', 'ARIMA', 'GAN', 'CoSMoS']
summary = pd.DataFrame(index=rows,
                       columns=['Mean', 'std', 'Skew', 'Kurtosis', 'DTW', 'Wass.'])

for i in fourier_stats.index.values.tolist():
    summary.loc['Original', i] = fourier_stats.loc[i, f'Orig. {model.data.name}'].round(2)
    summary.loc['Fourier', i] = fourier_stats.loc[i, f'Averages (+- Std Dev)']
summary.loc['Fourier', 'DTW'] = fourier_dtw.loc[0, 'Average (+- Std Dev)']
summary.loc['Fourier', 'Wass.'] = fourier_wass.loc[0, 'Average (+- Std Dev)']

for i in ARIMA_stats.index.values.tolist():
    summary.loc['ARIMA', i] = ARIMA_stats.loc[i, f'Averages (+- Std Dev)']
summary.loc['ARIMA', 'DTW'] = ARIMA_dtw.loc[0, 'Average (+- Std Dev)']
summary.loc['ARIMA', 'Wass.'] = ARIMA_wass.loc[0, 'Average (+- Std Dev)']

for i in GAN_stats.index.values.tolist():
    summary.loc['GAN', i] = GAN_stats.loc[i, f'Averages (+- Std Dev)']
summary.loc['GAN', 'DTW'] = GAN_dtw.loc[0, 'Average (+- Std Dev)']
summary.loc['GAN', 'Wass.'] = GAN_wass.loc[0, 'Average (+- Std Dev)']

for i in cosmos_stats.index.values.tolist():
    summary.loc['CoSMoS', i] = cosmos_stats.loc[i, f'Averages (+- Std Dev)']
summary.loc['CoSMoS', 'DTW'] = cosmos_dtw.loc[0, 'Average (+- Std Dev)']
summary.loc['CoSMoS', 'Wass.'] = cosmos_wass.loc[0, 'Average (+- Std Dev)']

summary.to_csv(os.path.join(export_dir, f'Summary - {data_name}.csv'))

#%%
# Load Time Series Data
import random
rand = random.randint(0, 999) # Pick out random synth series index

time_series_data = [model.data.values, synth_data_fourier.iloc[:,rand], synth_data_ARIMA.iloc[:,rand],
                    synth_data_GAN.iloc[:,rand], synth_data_cosmos.iloc[:,rand]]
series_names = ['Original', 'Fourier', 'ARIMA', 'GAN', 'CoSMoS']

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