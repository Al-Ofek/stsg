import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

#%%
data_name = 'wave_height'

# Set the working directory to the location of the data
export_dir = f'{os.getcwd()}\\{data_name} generated data'

# Load the data
data = pd.read_csv(os.path.join(os.getcwd(), f'{data_name}.csv'), dtype=float)
fourier_stats = pd.read_csv(os.path.join(export_dir, f'{data_name}_fourier_stats.csv'))
fourier_dtw = pd.read_csv(os.path.join(export_dir, f'{data_name}_fourier_dtw.csv'))
fourier_wass = pd.read_csv(os.path.join(export_dir, f'{data_name}_fourier_wass.csv'))
ARIMA_stats = pd.read_csv(os.path.join(export_dir,f'{data_name}_ARIMA_stats.csv'))
ARIMA_dtw = pd.read_csv(os.path.join(export_dir, f'{data_name}_ARIMA_dtw.csv'))
ARIMA_wass = pd.read_csv(os.path.join(export_dir, f'{data_name}_ARIMA_wass.csv'))
cosmos_stats = pd.read_csv(os.path.join(export_dir, f'{data_name}_cosmos_stats.csv'))
cosmos_dtw = pd.read_csv(os.path.join(export_dir, f'{data_name}_cosmos_dtw.csv'))
cosmos_wass = pd.read_csv(os.path.join(export_dir, 'f{data_name}_cosmos_wass.csv'))
GAN_stats = pd.read_csv(os.path.join(export_dir, 'f{data_name}_GAN_stats.csv'))
GAN_dtw = pd.read_csv(os.path.join(export_dir, f'{data_name}_GAN_dtw.csv'))
GAN_wass = pd.read_csv(os.path.join(export_dir, f'{data_name}_GAN_wass.csv'))

#%% Summary
rows = ['Original', 'Fourier', 'ARIMA', 'GAN', 'CoSMoS']
summary = pd.DataFrame(index=rows,
                       columns=['Mean', 'std', 'Skew', 'Kurtosis', 'DTW', 'Wass.'])

for i in fourier_stats.index.values.tolist():
    summary.loc['Original', i] = fourier_stats.loc[i, f'Orig. {model.data.name}']
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


#%% Comparison Plot
# Load Time Series Data
import random
rand = random.randint(0, 999) # Pick out random synth series index

time_series_data = [model.data.iloc[:,0], synth_data_fourier.iloc[:,rand], synth_data_ARIMA.iloc[:,rand],
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
