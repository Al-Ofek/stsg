# Sets file folder as working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

#Imports STSG Class
from stsg import STSG

#%% Urban Demand Dataset

# Import data as dataframe
data = pd.read_csv('urban_demand.csv')

# Sets the data start/end times
start = '01/07/2020 00:00'
end = '14/07/2020 00:00'
# You may also specify the format of the datetime. The one specfied here is the default.
# datetime_format = '%d/%m/%Y %H:%M'

# Can also input index values:
# start = 0
# end = 500

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, data_start=start, data_end=end)


# Generate data using ARIMA
# Hyperparameters for ARIMA were obtained using ACF plot:
# import statsmodels.api as sm
# sm.graphics.tsa.plot_acf(model.data.values, lags = 40)
# synth_data_arima = model.gen_ARIMA(p=25, d=1, q=25, n_samples=3)
# Plot original + synthesized data
# model.plot(s_data=synth_data_arima, title='Urban Demand, ARIMA')

# Generate data using Fourier
synth_data_fourier = model.gen_fourier(keep_phases=3, n_samples=5, non_negative=True)
# Plot original + synthesized data
model.plot(s_data=synth_data_fourier, title='Urban Demand, fourier')
#%% PO dataset
# Import data as dataframe
import pandas as pd
data = pd.read_csv('PO.csv')

# Sets the data start/end times
start = '01/01/2024 00:05'
end = '08/01/2024 00:00'

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, data_start=start, data_end=end)

# Generate data using ARIMA
# Hyperparameters for ARIMA were obtained using ACF and PACF plots:
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(model.data.values, lags=50)

# sm.graphics.tsa.plot_acf(stsg.data.values, lags = 40)
# synth_data_arima = model.gen_ARIMA(p=25, d=1, q=25, n_samples=3)

# Plot original + synthesized data
# model.plot(s_data=synth_data_arima)

# Generate data using Fourier
synth_data_fourier = model.gen_fourier(keep_phases=3, n_samples=1)
# Plot original + synthesized data
model.plot(s_data=synth_data_fourier, title='PO 2.5, fourier')

#%% NO dataset
# Import data as dataframe
import pandas as pd
data = pd.read_csv('NO.csv')

datetime_format = '%d/%m/%Y'

# Create an instance of the class, using the selected subsection of the data
model = STSG(data, time_fmt=datetime_format)

# Generate data using ARIMA
# synth_data_arima = model.gen_ARIMA(p=25, d=1, q=25, n_samples=3)
# Plot original + synthesized data
# model.plot(s_data=synth_data_arima, title='NO, ARIMA')

# Generate data using Fourier
NO_synth_data_fourier = model.gen_fourier(keep_phases=100, n_samples=5, non_negative=True)
# Plot original + synthesized data
model.plot(s_data=NO_synth_data_fourier, title='NO, fourier')

#%% GAN Model

# Train the CTGAN model.
# Here shown using hyperparamters derived from trial and error.
# gan_synthesizer = model.train_GAN(epochs=1000, batch_size=200,
                                  # gen_lr=2e-5, disc_lr=2e-5,
                                  # filename='GAN_model')

# Optionally, you may load a previously trained model to use:
# from sdv.single_table import CTGANSynthesizer
# gan_synthesizer = CTGANSynthesizer.load(filepath='GAN_model.pkl')

# Generate data using CTGAN
synth_data_GAN = model.gen_GAN(n_samples=7,trained_model=gan_synthesizer)

# Plot original + synthesized data
model.plot(s_data=synth_data_GAN, title='GAN')

#%% Metrics

#Calculate DTW score and Wasserstein dist. for synthetic data.
#By default compares to model data, but one may specify different ref_data
dtw = model.dtw_score(s_data=synth_data_fourier)
wass = model.wass_dist(s_data=synth_data_fourier)

#%% Statistical Analysis

NO_stats = model.calculate_statistics(NO_synth_data_fourier)

# n_samples = len(NO_synth_data_fourier.axes[1])

# for i in range(n_samples):
#     synth_data = NO_synth_data_fourier[f'NO [µg/m³]_{i+1}']
#     mean = statistics.mean(synth_data)
#     std = statistics.pstdev(synth_data)
#     skew = stats.skew(synth_data)
#     kurtosis = stats.kurtosis(synth_data)
    