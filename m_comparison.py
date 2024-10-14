# Sets file folder as working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

#Imports STSG Class
from stsg import STSG

import matplotlib.pyplot as plt
#%% Urban Demand

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

# Create an instance of the class, using the selected subsection of the data
ud = STSG(data, data_start=start, data_end=end)

# ud_sum = ud.fourier_compare_m()

print(ud_sum)

#%% NO

# Import data as dataframe
data = pd.read_csv('NO.csv')

data_name = 'NO'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# # Sets the data start/end times
# start = '01/07/2020 00:00'
# end = '01/08/2020 00:00'

# number of synthetic plots to show in the background of all graphs
n_plots = 100

# Create an instance of the class, using the selected subsection of the data
no = STSG(data, time_fmt='%d/%m/%Y')

# no_sum = no.fourier_compare_m(dec_places=3, non_negative= 'abs')
# print(no_sum)

#%% PM 2.5
# Import data as dataframe
data = pd.read_csv('PM.csv')

data_name = 'PM'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# # Sets the data start/end times
# start = '01/07/2020 00:00'
# end = '01/08/2020 00:00'

# number of synthetic plots to show in the background of all graphs
n_plots = 100

# Create an instance of the class, using the selected subsection of the data
pm = STSG(data)

# pm_sum = pm.fourier_compare_m()
# print(pm_sum)

#%% u

# Import data as dataframe
data = pd.read_csv('u.csv', dtype=float)

data_name = 'u'

# Creates export directory
export_dir = f'{os.getcwd()}\\{data_name} generated data'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# number of synthetic plots to show in the background of all graphs
n_plots = 100

# Create an instance of the class, using the selected subsection of the data
u = STSG(data, data_start=0, data_end=2000, time_data=False)

# u_sum = u.fourier_compare_m()
# print(u_sum)

#%% wave height

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
wh = STSG(data, data_start=0, data_end=2000, time_data=False)

# wh_sum = wh.fourier_compare_m(dec_places=dec_places)
# print(wh_sum)

#%% Summary of all datasets

combined_summary = pd.concat([ud_sum, no_sum, pm_sum, u_sum,wh_sum], axis=0)

combined_summary.to_csv('Summary of m comparison.csv')

#%%


def plot_all_comparisons(instances, m_values=[3, 40, 100], n_samples=1000, n_plots=100, 
                         dec_places=2, non_negative=False):
    """
    Plots comparison plots for multiple instances of the STSG class on a single figure.
    
    Parameters:
    - instances: List of STSG class instances (e.g., [ud, no, pm, u, wh]).
    - m_values: List of m values to use in Fourier synthesis.
    - n_samples: Number of samples to generate.
    - n_plots: Number of plots to show.
    - dec_places: Decimal places for rounding in statistics.
    - non_negative: Boolean flag to keep synthesized data non-negative.
    """
    
    num_instances = len(instances)
    fig, axs = plt.subplots(num_instances, 3, figsize=(15, 5 * num_instances), sharex=True)
    
    if num_instances == 1:
        axs = [axs]  # Make sure axs is always a list of subplots, even with one instance
    
    for i, instance in enumerate(instances):
        summary = instance.fourier_compare_m(m_values=m_values, n_samples=n_samples, 
                                             n_plots=n_plots, dec_places=dec_places, 
                                             non_negative=non_negative, axs=axs[i])
        for ax in axs[i]:
            ax.set_ylim(axs[0][0].get_ylim())  # Ensure y-axis is the same across all subplots
    
    plt.tight_layout()
    plt.show()

# Usage example:
# plot_all_comparisons([ud, no, pm, u, wh])

#%%

def plot_multiple_fourier_by_instance(instances, m_values=[3, 40, 100], n_samples=100, non_negative=False):
    n_instances = len(instances)
    n_m_values = len(m_values)
    
    # fig_width = 433.62/72.27
    # fig_height = fig_width * 1
    
    fig_width = 12
    fig_height = 10
    
    fig, axs = plt.subplots(n_instances, n_m_values, figsize=(fig_width, fig_height), sharex='row', sharey='row')
    
    for i, instance in enumerate(instances):
        for j, m in enumerate(m_values):
            if instance == no:
                s_fourier = instance.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative='abs')
            else:
                s_fourier = instance.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative=False)
            for col in s_fourier:
                axs[i, j].plot(s_fourier[col].values, color='#b6dfe9')
            axs[i, j].plot(instance.data, color='#3D8DFF', linewidth=2, label='Original Data')
            axs[i, j].set_title(f'$m = {m}$' if i == 0 else "")
            if instance.time_data:
                x_ticks = np.linspace(start=0, stop=len(instance.data)-1, num=4).astype(int)
                time_values = pd.Series(pd.to_datetime(instance.data.index[x_ticks], format=instance.time_fmt))
                x_labels = time_values.dt.strftime('%d/%m/%y')
                axs[i, j].set_xticks(x_ticks)
                axs[i, j].set_xticklabels(x_labels)
            if j == 0:
                axs[i, j].set_ylabel(instance.data.name)

    plt.tight_layout()
    plt.show()

# Example usage with dummy instances
instances = [ud, no, pm, u, wh]
plot_multiple_fourier_by_instance(instances)


#%%
def plot_multiple_fourier_by_instance(instances, m_values=[3, 40, 100], n_samples=20, non_negative=False):
    n_instances = len(instances)
    n_m_values = len(m_values)
    
    fig_width = 433.62/72.27
    fig_height = fig_width * 1.4
    
    fig, axs = plt.subplots(n_instances, figsize=(fig_width, fig_height), sharex='row', sharey='row')
    
    colors = ['#41b6c4','#2c7fb8','#253494']
    alphas = [0.5, 0.5, 0.2]
    
    for i, instance in enumerate(instances):
        for j, m in enumerate(m_values):
            if instance == no:
                s_fourier = instance.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative='abs')
            else:
                s_fourier = instance.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative=False)
            for col in s_fourier:
                axs[i].plot(s_fourier[col].values, color=colors[j], alpha=alphas[j])
            axs[i].plot(instance.data, color='#253494', linewidth=2, label='Original Data')

            if instance.time_data:
                x_ticks = np.linspace(start=0, stop=len(instance.data)-1, num=4).astype(int)
                time_values = pd.Series(pd.to_datetime(instance.data.index[x_ticks], format=instance.time_fmt))
                x_labels = time_values.dt.strftime('%d/%m/%y')
                axs[i].set_xticks(x_ticks)
                axs[i].set_xticklabels(x_labels)
            if j == 0:
                axs[i].set_ylabel(instance.data.name)

    plt.tight_layout()
    plt.show()

# Example usage with dummy instances
instances = [ud, no, pm, u, wh]
plot_multiple_fourier_by_instance(instances)
