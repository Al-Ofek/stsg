# Sets file folder as working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

#Imports STSG Class
from stsg import STSG

# Ignore PreformanceWarning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

#%% Model Creation

# Import data as dataframe
data = pd.read_csv('wave_height.csv', dtype=float)

data_name = 'wave_height'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# number of synthetic plots to show in the background of all graphs
n_plots = 100

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, data_start=0, data_end=2000, time_data=False)

#%% Fourier
# Generate data using Fourier
synth_data = model.gen_fourier(keep_phases=3, n_samples=1)
model.plot(s_data=synth_data.iloc[:,:100], title='Wave Height, Fourier Method')

#%% Histogram (bar chart)
bins = np.histogram_bin_edges(model.data.values, bins=10) #, range=) #range needs to be min/max of both synthetic and reference data

plt.figure(figsize=(8,4), dpi=300)
sns.histplot(synth_data, stat='probability', bins=bins, alpha=1, common_norm=False,
              legend=None, edgecolor=None)
sns.histplot(model.data, stat='probability', bins=bins, alpha=0, common_norm=False,
              legend=None)
plt.title('Overlayed Histograms')
plt.xlabel('Value')
plt.show()

#%% Histogram v.2 (smooth)
plt.figure(figsize=(8,4), dpi=300)
sns.kdeplot(synth_data, alpha=0.8, legend=None)
plt.title('Overlayed Histograms')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

#%%
# Set up the plot
plt.figure(figsize=(8,4), dpi=300)

lags = 20

# Plot ACF for each time series
plot_acf(synth_data.iloc[:,0], lags=lags, alpha=0.05, ax=plt.gca(), color='orange')
# plot_acf(data3, lags=20, alpha=0.05, ax=plt.gca(), color='green')
# plot_acf(model.data.values, lags=lags, alpha=0.05, ax=plt.gca(), color='blue')
# Customize the plot
plt.title('ACF')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.ylim(top=1.3,bottom=-1.3)
plt.legend(['Orig. Time Series'])
plt.show()


#%% PLOTTING EXAMPLES FROM CHAT.GPT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

# Step 2: Generate or Load Time Series Data
# For demonstration, we will generate 5 random time series
np.random.seed(0)
time_series_data = [np.cumsum(np.random.randn(100)) for _ in range(5)]
series_names = [f'Series {i+1}' for i in range(5)]

# Step 3: Create Subplots for Time Series, Distribution, and ACF
fig, axes = plt.subplots(5, 3, figsize=(15, 15), gridspec_kw={'width_ratios': [2, 1, 1]})

# Step 4: Plot Time Series
for i, ax in enumerate(axes):
    # Time series plot
    axes[i, 0].plot(time_series_data[i])
    axes[i, 0].set_title(series_names[i])
    axes[i, 0].set_ylabel(series_names[i])
    
    # Step 5: Plot Distribution
    sns.histplot(time_series_data[i], ax=axes[i, 1])
    axes[i, 1].set_title('Distribution')
    
    # Step 6: Plot ACF
    plot_acf(time_series_data[i], ax=axes[i, 2], title='ACF')

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a figure with three vertically stacked subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Plot data
axs[0].plot(x, y1)
axs[0].set_title('Sin(x)')

axs[1].plot(x, y2)
axs[1].set_title('Cos(x)')

axs[2].plot(x, y3)
axs[2].set_title('Tan(x)')
axs[2].set_ylim(-10, 10)  # Limit y-axis for better visualization

# Set common labels
plt.xlabel('X values')
fig.text(0.04, 0.5, 'Y values', va='center', rotation='vertical')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
#%%

model.plot_fourier(m_values=[3,20,100])