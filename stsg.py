import numpy as np
import pandas as pd
import scipy
import math
import random
import matplotlib.pyplot as plt
from typing import Union
from math import pi
import warnings
import statistics

#ARIMA
from statsmodels.tsa.arima.model import ARIMA

#Metrics
from tslearn.metrics import dtw
from scipy.stats import wasserstein_distance

from sklearn.preprocessing import MinMaxScaler

class STSG:
    """" Synthetic Time Series Generator
    
    Class for creating synthetic time series similar to a real-world time 
    series, using several different methods - ARIMA, GAN, and randomizing
    Fourier coefficients.
    
    Parameters
    ----------
    df_base_data_path: str or DataFrame
        Recieves path to a .csv data file as str, or alternatively
        an already created pandas dataframe object.
    
    data_start/data_end: int or str
        Selects what part of the time series to import. 
        If both are None, all the data is imported.
        To select a certain time window, use int 
        (eg. data_start=0, data_end=100 select the first 100 entries).
        Alternatively if datetime data is present, it is possible to
        specify start/end times as str (eg. data_start = '01/01/2020 00:00'
                                            data_end = '02/01/2020 00:00').
        Note that the string should be translatebale to a datetime object.
    
    time_data: bool
        Imported data is assumed to have a datetime column at 0,
        and time series values at 1. To use without datetime data set to 
        False and import file with single column.
        
    time_fmt: str
        The format of the time data, if exisiting. For example '%d/%m/%Y %H:%M',
        which is set as default. This is required to prevent issues with pandas.
        For more details see:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    """
    def __init__(self, df_base_data_path : Union[pd.DataFrame, str],
                 time_data=True, data_start=None, data_end=None, time_fmt='%d/%m/%Y %H:%M'):
        self.data_start = data_start
        self.data_end = data_end
        self.time_data = time_data
        self.time_fmt = time_fmt
        
        if isinstance(df_base_data_path, pd.DataFrame):
            #pd.DataFrame constructor
            self.base_data_path = None
            if self.time_data is True: #Checks for date/time column to use as index
                df_base_data_path.set_index(df_base_data_path.columns[0], inplace=True)
            self.data_loader(df_base_data_path)
        else:
            #str constructor
            self.base_data_path = df_base_data_path
            if self.time_data is True:
                self.data_loader(pd.read_csv(self.base_data_path, index_col=0))
            else:
                self.data_loader(pd.read_csv(self.base_data_path))

                
    def data_loader(self, df):
        #this function should not be called by the user
        #it puts the data into self.test, and the selected week into self.data
        if self.time_data == True:
            df = df.iloc[:,0]
            self.test = df.copy()
            self.data = df.copy()
            self.test.index = pd.to_datetime(self.test.index, format=self.time_fmt)
            
            if self.data_start is None and self.data_end is None:
                pass
            # runs if start/end is int
            elif all(isinstance(i, int) for i in [self.data_start, self.data_end]):
                self.data = self.data.iloc[self.data_start:self.data_end]
            # runs if start/end is string
            elif all(isinstance(i, str) for i in [self.data_start, self.data_end]):
                # self.data_start = pd.to_datetime(self.data_start, format='%d/%m/%Y %H:%M')
                # self.data_end = pd.to_datetime(self.data_end, format='%d/%m/%Y %H:%M')
                self.data = self.data.loc[self.data_start:self.data_end].astype(float)
        else:
            self.data = df[:]            
            if self.data_start is None and self.data_end is None:
                pass
            # runs if start/end is int
            elif all(isinstance(i, int) for i in [self.data_start, self.data_end]):
                self.data = self.data.iloc[self.data_start:self.data_end]
            
            self.data.name = df.columns[0]

                
    def gen_fourier(self, keep_phases=1, n_samples=1, non_negative : str = None):
        """Generates time series based on original data by using a discrete
        Fourier transform, and randomizing a set number of phases.
        
        Parameters
        ----------
        keep_phases : int
            number of phases that will not be randomized. Accepts non-zero
            positive integers. For example, keep_phases=3 keeps phases 0,1,2. 
            
        n_samples : int
            Number of synthetic samples to be generated, each of the same
            length as the original data.
        
        non_negative : bool
            To be used in case of a time series for which negative values
            make no physical sense. If True, the function generates a series,
            and checks if it contains negative values. If they exist, a new series
            is generated, and if the new values are non-negative where the orig.
            values were negative, they are used instead. This process continues
            until all values are non-negative.
        
        Returns
        -------
        DataFrame
            dataframe containing synthesized data samples in columns.
            Columns are named by the column name of the original data.
            eg. 'Demand' in the original file will make columns 'Demand_1',
            'Demand_2' etc.
        """
        
        def gen_single_series(series):
                nonlocal keep_phases
                if keep_phases <= 0:
                   keep_phases= 1
                
                #This is the REAL fourier transform for REAL functions, it will
                #output the fourier coefficients up to n/2 as a real sequence,
                #that determine the rest of the coefficients.
                F = scipy.fft.rfft(series)

                #this loop runs through the range from keep_phases+1 to
                # (\frac{n}{2} - 1) (inclusive) if n is even, and 
                # \frac{n-1}{2} (inclusive) if n is odd.
                for i in range(keep_phases, (len(self.data.values)+1) // 2):  # Changed >>1 to // 2 to improve readability
                    r = math.sqrt(F[i].real*F[i].real + F[i].imag*F[i].imag)
                    ang = (random.random() - 0.5) * 2*pi
                    F[i] = r*math.cos(ang) + r*math.sin(ang)*1j
                
                #Runs if $\frac{n}{2}$th coeffient is not being preserved.
                #There are only two possibilities for randomizing it, as it needs
                # to be both real and have the same absolute value
                if len(self.data.values) % 2 == 0: # Changed &1 to %2 to improve readability
                    if keep_phases <= (len(self.data.values) // 2):
                        if random.getrandbits(1) != 0: 
                            F[len(self.data.values) // 2] *= -1 
                            
                return scipy.fft.irfft(F,n=len(self.data.values))
            
        d = {}
        
        og_series = self.data.values.squeeze()
        
        if non_negative == 'orig.':
            for k in range(n_samples):
                synth_series = gen_single_series(og_series)
                while np.min(synth_series) < 0:
                    alt_series = gen_single_series(synth_series) #generates from current synth series
                    for i, x in enumerate(synth_series):
                        if x < 0:
                            synth_series[i] = alt_series[i]
                d[self.data.name+'_'+str(k+1)] = synth_series
        
        elif non_negative == 'abs':
            for k in range(n_samples):
                d[self.data.name+'_'+str(k+1)] = np.abs(gen_single_series(og_series))
        elif non_negative == 'shift':
            for k in range(n_samples):
                s = gen_single_series(og_series)
                min_val, max_val = np.min(s), np.max(s)
                if min_val < 0:
                    scaler = MinMaxScaler(feature_range=(0, max_val))
                    s = s + abs(min_val)
                    s = scaler.fit_transform(s.reshape(-1,1))[:,0]
                d[self.data.name+'_'+str(k+1)] = s
        else:
            for k in range(n_samples):
                d[self.data.name+'_'+str(k+1)] = gen_single_series(og_series)
                
        return pd.DataFrame(d)

    def gen_ARIMA(self, p=3, d=1, q=3, n_samples=1):
        """Generate ARMA model with specified hyperparameters using the
        statsmodels library.
        
        Parameters
        ----------
        p: int
            Order of the autoregressive (AR) part of the model.
        d:
            Degree of first differencing (I) in the model.
            Set d=0 (used as default) to get an ARMA model.
        q: int
            Order of the moving average (MA) part of the model.
        
        n_samples : int
            Number of synthetic samples to be generated, each of the same
            length as the original data.
        
        Returns
        -------
        DataFrame
            dataframe containing synthesized data samples in columns.
            Columns are named by the column name of the original data.
            eg. 'Demand' in the original file will make columns 'Demand_1',
            'Demand_2' etc.
        """

        mod = ARIMA(endog=self.data.values,order=(p,d,q)).fit()
        s = {}
        for k in range(n_samples):
            sim = mod.simulate(nsimulations=len(self.data))
            s[self.data.name + '_' + str(k+1)] = sim

        return pd.DataFrame(s)

    
    def plot(self, s_data : pd.DataFrame, ref_data : pd.DataFrame = None ,title=None, filename=None):
        """ Plots the original and synthsized data on the same graph.
        
        Properties
        ----------
        s_data: DataFrame
            Synthesized data to plot, generated by one of the class methods 
            (gen_fourier, etc.). Expecting dataframe with time series values in
            columns, each of the same length as the original data.
            
        ref_data: DataFrame
            Real-world data to plot as reference to the synthesized time series.
            By default, set to be the original data set used in class definition.
            If using a different set of data, it should be a dataframe 
            containing a single column, of the same length as s_data.
        
        title: str
            Optional. Title for the generated graph.
        
        filename: str
            File name for the saved graph.
            If not specified, graph will not be saved!
            
        """
        if isinstance(ref_data, pd.DataFrame):
            pass
        else:
            ref_data = self.data.values.squeeze().astype(float)
        smpl = s_data
        
        fig = plt.figure(figsize=(8,3), dpi=300)
        if isinstance(title, str):
            plt.title(str(title))
        plt.ylabel(str(self.data.name))
        
        y_ticks = np.linspace(start=min([min(ref_data),min(s_data.min())]),
                              stop=max([max(ref_data),max(s_data.max())]), num=6)
        plt.yticks(ticks=y_ticks)
        
        # Uses datetime for x axis labels, if available
        if self.time_data == True:
            x_ticks = np.linspace(start=0,stop=len(ref_data)-1, num=6).astype(int)
            time_values = pd.Series(pd.to_datetime(self.data.index[x_ticks], format=self.time_fmt))
            x_labels = time_values.dt.strftime('%d/%m/%y \n %H:%M')
            plt.xticks(ticks=x_ticks, labels=x_labels)
        else:
            plt.xlabel('Index')
        
        for col in smpl.columns:
            plt.plot(smpl[col].values, color='#C5E3EA')
        plt.plot(ref_data, color='#3D8DFF', linewidth=2, label='Original Data')
        
        if isinstance(filename, str):
            fig.savefig(filename, bbox_inches='tight')
            
        plt.show()
        
    def plot_fourier(self, m_values : list = [1, 15, 100], n_samples : int = 100,
                     non_negative : bool = False):
        """
        Generates 3 plots in a single figure, with the orig. series in the foreground,
        and a multitude of synthetic (Fourier generated) series in the background.
        
        Parameters
        ----------
        m_values : list, optional
            A list containing the number of coefficents to retain in generating the series
            for each plot. The default is [1, 15, 100].
        n_samples : int, optional
            Number of synthetic samples in each plot. The default is 100.
        non_negative: bool
            Same as gen_fourier. By default False.
            
        Returns
        -------
        None.

        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)    
        s_fourier = []
        for i,m in enumerate(m_values):
            s_fourier.append(self.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative=non_negative))
            for col in s_fourier[i]:
                axs[i].plot(s_fourier[i].values, color='#C5E3EA')
            axs[i].plot(self.data, color='#3D8DFF', linewidth=2, label='Original Data')
            axs[i].set_title(f'$m = {m}$')
            if self.time_data == True:
                x_ticks = np.linspace(start=0, stop=len(self.data)-1, num=6).astype(int)
                time_values = pd.Series(pd.to_datetime(self.data.index[x_ticks], format=self.time_fmt))
                x_labels = time_values.dt.strftime('%d/%m/%y')
                axs[i].set_xticks(x_ticks)
                if i == 2:  # Only label the x-axis on the bottom plot
                    axs[i].set_xticklabels(x_labels)
            else:
                if i==2:
                    axs[i].set_xlabel('Index')
            if i==1:
                axs[i].set_ylabel(self.data.name)
        
        plt.tight_layout()
        plt.show()

    def fourier_compare_m(self, m_values : list = [3, 40, 100], n_samples=1000,
                          n_plots=100, dec_places : int = 2, non_negative : bool = False):    
        
        # Initialize summary DataFrame
        summary = pd.DataFrame(index=[f'Original {self.data.name}'] + [f'm={m}' for m in m_values],
                               columns=['Mean', 'std', 'Skew', 'Kurtosis', 'DTW', 'Wass.'])
        
        # Calculate statistics for the original data
        original_stats = self.calculate_statistics(dec_places=dec_places)
        for stat in original_stats.index.values.tolist():
            summary.loc[f'Original {self.data.name}', stat] = original_stats.loc[stat, f'Orig. {self.data.name}']
        summary.loc[f'Original {self.data.name}', 'DTW'] = '-'  # DTW does not apply to original data
        summary.loc[f'Original {self.data.name}', 'Wass.'] = '-'  # Wasserstein distance does not apply to original data

        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)    
        s_fourier = []

        for i,m in enumerate(m_values):    
            # Generate data using Fourier
            synth_data_fourier = self.gen_fourier(keep_phases=m, n_samples=n_samples, non_negative=non_negative)
            s_fourier.append(synth_data_fourier.iloc[:,:n_plots])
                
            # similarity scores
            fourier_dtw = self.dtw_score(synth_data_fourier,dec_places=dec_places)
            fourier_wass = self.wass_dist(synth_data_fourier,dec_places=dec_places)
            
            # statistics
            fourier_stats = self.calculate_statistics(synth_data_fourier,dec_places=dec_places)
            
            # Populate summary table
            for stat in fourier_stats.index.values.tolist():
                summary.loc[f'm={m}', stat] = fourier_stats.loc[stat, f'Averages (+- Std Dev)']
            summary.loc[f'm={m}', 'DTW'] = fourier_dtw.loc[0, 'Average (+- Std Dev)']
            summary.loc[f'm={m}', 'Wass.'] = fourier_wass.loc[0, 'Average (+- Std Dev)']
            
            # plot
            for col in s_fourier[i]:
                axs[i].plot(s_fourier[i].values, color='#C5E3EA')
            axs[i].plot(self.data, color='#3D8DFF', linewidth=2, label='Original Data')
            axs[i].set_title(f'$m = {m}$')
            if self.time_data == True:
                x_ticks = np.linspace(start=0, stop=len(self.data)-1, num=6).astype(int)
                time_values = pd.Series(pd.to_datetime(self.data.index[x_ticks], format=self.time_fmt))
                x_labels = time_values.dt.strftime('%d/%m/%y')
                axs[i].set_xticks(x_ticks)
                if i == 2:  # Only label the x-axis on the bottom plot
                    axs[i].set_xticklabels(x_labels)
            else:
                if i==2:
                    axs[i].set_xlabel('Index')
            if i==1:
                axs[i].set_ylabel(self.data.name)
        
        plt.tight_layout()
        plt.show()
        
        return summary
        
    def dtw_score(self, s_data : pd.DataFrame, ref_data : pd.DataFrame = None,
                  dec_places : int = 2):
        """ Calculates DTW similarity score between orig. and synthetic series.
        
        Implemented using tslearn library:
            tslearn.readthedocs.io
            
        Note: The score is not normalized, so to get a proper comparision one
        should use series of the same length, and with similar orders of
        magnitude.
        
        Parameters
        ----------
        s_data: DataFrame
            Synthetic data to compare. Should be of the same length as original
            data.
            
        ref_data: DataFrame
            Reference data to compare the synthetic series to.
            By default, set to be the original data set used in class definition.
            If using a different set of data, it should be a dataframe 
            containing a single column, of the same length as s_data.
            
        dec_places: int
            Number of decimal places to display (in the average & std dev).
        
        Returns:
        --------
        DataFrame
            First has the average value and std, then the calculated
            DTW score in columns.
        """
        if isinstance(ref_data, pd.DataFrame):
            pass
        else:
            ref_data = self.data.values
        smpl = s_data
        
        score = {}
        for col in smpl.columns:
            score[col] = dtw(ref_data, smpl[col])
        
        dtw_data = pd.DataFrame(score, index=[0])
        
        average_dtw = statistics.mean(dtw_data.iloc[0])
        std_dtw = statistics.pstdev(dtw_data.iloc[0])
        
        dtw_data.insert(0,'Average (+- Std Dev)',
                        f'{average_dtw:.{dec_places}f} +- {std_dtw:.{dec_places}f}')
                
        return dtw_data
    
    def wass_dist(self, s_data : pd.DataFrame, ref_data : pd.DataFrame = None,
                  dec_places : int = 2):
            """ Calculates Wasserstein distance between orig. and synthetic series.
        
        Implemented using scipy.stats.
         
        Note: The score is not normalized, so to get a proper comparision one
        should use series of the same length, and with similar orders of
        magnitude.
        
        Parameters
        ----------
        s_data: DataFrame
            Synthetic data to compare. Should be of the same length as original
            data.
            
        ref_data: DataFrame
            Reference data to compare the synthetic series to.
            By default, set to be the original data set used in class definition.
            If using a different set of data, it should be a dataframe 
            containing a single column, of the same length as s_data.
    
        dec_places: int
            Number of decimal places to display (in the average & std dev).
        
        Returns:
        --------
        DataFrame
            First has the average value and std, then the calculated
            Wasserstein dist. in columns.
        """
            if isinstance(ref_data, pd.DataFrame):
                pass
            else:
                ref_data = self.data.values.squeeze()
                
            smpl = s_data
        
            score = {}
            for col in smpl.columns:
                score[col] = wasserstein_distance(ref_data, smpl[col])
            
            wass_data = pd.DataFrame(score, index=[0])
            average_wass = statistics.mean(wass_data.iloc[0])
            std_wass = statistics.pstdev(wass_data.iloc[0])

            wass_data.insert(0,'Average (+- Std Dev)',
                             f'{average_wass:.{dec_places}f} +- {std_wass:.{dec_places}f}')
                                
            return wass_data
            
    def calculate_statistics(self, s_data: pd.DataFrame = None,
                             ref_data: pd.DataFrame = None, dec_places: int = 2):
        """
        Calculates the statistical moments of the synthetic and original data.
    
        Parameters
        ----------
        s_data: DataFrame, optional
            Synthetic data. Should be of the same length as original data.
            If None, only the statistics for the original data will be returned.
    
        ref_data: DataFrame, optional
            The original time series data. 
            By default, set to be the original data set used in class definition.
            If using a different set of data, it should be a dataframe 
            containing a single column, of the same length as s_data.
            
        dec_places: int
            Number of decimal places to display (in the average & std dev).
            
        Returns:
        --------
        DataFrame
            The first 2 columns contain the averages of the statistical moments
            and the std. Next is the data for the original series, and the rest are
            columns for each time series, containing the calculated values.
        """
        
        if isinstance(ref_data, pd.DataFrame):
            pass
        else:
            ref_data = self.data.values
        
        data_stats = pd.DataFrame(index=['Mean', 'std', 'Skew', 'Kurtosis'])
        
        # Calculate statistics for original data
        series_data = ref_data.squeeze()
        data_stats.loc['Mean', f'Orig. {self.data.name}'] = statistics.mean(series_data)
        data_stats.loc['std', f'Orig. {self.data.name}'] = statistics.pstdev(series_data)
        data_stats.loc['Skew', f'Orig. {self.data.name}'] = scipy.stats.skew(series_data)
        data_stats.loc['Kurtosis', f'Orig. {self.data.name}'] = scipy.stats.kurtosis(series_data)
        
        if s_data is not None:
            n_samples = len(s_data.axes[1])
            
            for k in range(1, n_samples + 1):
                series_data = s_data[f'{self.data.name}_{k}']
                data_stats.loc['Mean', f'{self.data.name}_{k}'] = statistics.mean(series_data)
                data_stats.loc['std', f'{self.data.name}_{k}'] = statistics.pstdev(series_data)
                data_stats.loc['Skew', f'{self.data.name}_{k}'] = scipy.stats.skew(series_data)
                data_stats.loc['Kurtosis', f'{self.data.name}_{k}'] = scipy.stats.kurtosis(series_data)
            
            # Calculate averages and standard deviations for the synthetic data
            avg_mean = statistics.mean(data_stats.loc['Mean'].iloc[1:])
            avg_std = statistics.mean(data_stats.loc['std'].iloc[1:])
            avg_skew = statistics.mean(data_stats.loc['Skew'].iloc[1:])
            avg_kurtosis = statistics.mean(data_stats.loc['Kurtosis'].iloc[1:])
            
            std_mean = statistics.pstdev(data_stats.loc['Mean'].iloc[1:])
            std_std = statistics.pstdev(data_stats.loc['std'].iloc[1:])
            std_skew = statistics.pstdev(data_stats.loc['Skew'].iloc[1:])
            std_kurtosis = statistics.pstdev(data_stats.loc['Kurtosis'].iloc[1:])
            
            averages = [f'{avg_mean:.{dec_places}f} +- {std_mean:.{dec_places}f}',
                        f'{avg_std:.{dec_places}f} +- {std_std:.{dec_places}f}',
                        f'{avg_skew:.{dec_places}f} +- {std_skew:.{dec_places}f}',
                        f'{avg_kurtosis:.{dec_places}f} +- {std_kurtosis:.{dec_places}f}']
            data_stats.insert(0, 'Averages (+- Std Dev)', averages)
    
        return data_stats
