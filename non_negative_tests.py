# Sets file folder as working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

#Imports STSG Class
from stsg import STSG

import scipy
import scipy.special as sp
import matplotlib.pyplot as plt

# Ignore PreformanceWarning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#%%

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

no_sum = no.fourier_compare_m(n_samples=1000, n_plots=100, dec_places=3, non_negative= 'shift')
print(no_sum)
#%%
F_no = scipy.fft.rfft(no.data.values)
F_no_sqr = np.square(np.abs(F_no))
plt.plot(F_no_sqr)

#%%

def hermite_polynomial(n, x):
    """Generate the n-th Hermite polynomial evaluated at x."""
    return sp.hermite(n)(x)

def decompose_signal_to_hermite(signal, max_order=10):
    """
    Decompose the complex signal into Hermite polynomials up to a given order.

    :param signal: Complex signal (1D array).
    :param max_order: Maximum order of Hermite polynomials to use for decomposition.
    :return: Coefficients of the Hermite polynomial expansion.
    """
    N = len(signal)
    x = np.linspace(-1, 1, N)
    coefficients = np.zeros((max_order + 1,), dtype=complex)

    for n in range(max_order + 1):
        H_n = hermite_polynomial(n, x)
        coefficients[n] = np.dot(signal, H_n) / np.dot(H_n, H_n)

    return coefficients

def sum_elements_divisible_by_4(coefficients):
    """
    Sum the coefficients that are at indices divisible by 4.

    :param coefficients: Coefficients from the Hermite polynomial decomposition.
    :return: New signal formed by summing the selected coefficients.
    """
    return sum(coefficients[i] for i in range(0, len(coefficients), 4))

# Example usage
signal = np.random.randn(100) + 1j * np.random.randn(100)  # Example complex signal
coefficients = decompose_signal_to_hermite(F, max_order=20)
new_signal = sum_elements_divisible_by_4(coefficients)

print("New Signal:", new_signal)

#%%
import numpy as np
from numpy.polynomial.hermite import hermval,hermfit

def decompose_signal_to_hermite(signal, max_order=10):
    """
    Decompose the complex signal into Hermite polynomials up to a given order.

    :param signal: Complex signal (1D array).
    :param max_order: Maximum order of Hermite polynomials to use for decomposition.
    :return: Coefficients of the Hermite polynomial expansion.
    """
    N = len(signal)
    x = np.linspace(-1, 1, N)  # Sample points for evaluation
    coefficients = np.zeros((max_order + 1,), dtype=complex)

    for n in range(max_order + 1):
        H_n = np.polynomial.hermite.hermval(x, [0]*n + [1])  # Generate n-th Hermite polynomial
        coefficients[n] = np.dot(signal, H_n) / np.dot(H_n, H_n)

    return coefficients

def generate_signal_divisible_by_4(coefficients):
    """
    Generate a new signal using only the Hermite coefficients at indices divisible by 4.

    :param coefficients: Coefficients from the Hermite polynomial decomposition.
    :return: New signal formed by summing the selected coefficients.
    """
    new_coeffs = np.zeros_like(coefficients)
    new_coeffs[::4] = coefficients[::4]  # Keep only coefficients at indices divisible by 4
    return new_coeffs

def reconstruct_signal(coefficients, x):
    """
    Reconstruct the signal from Hermite polynomial coefficients.

    :param coefficients: Coefficients from the Hermite polynomial decomposition.
    :param x: Points where the Hermite polynomials are evaluated.
    :return: Reconstructed signal.
    """
    return hermval(x, coefficients)

# Example usage
F_no = scipy.fft.rfft(no.data.values)
signal = F_no  # Example complex signal
max_order = 100
x = np.arange(len(F_no))

# Decompose the signal into Hermite polynomials
# coefficients = decompose_signal_to_hermite(signal, max_order=max_order)
coefficients = hermfit(x,F_no, deg=max_order)

# Generate a new signal using only coefficients at indices divisible by 4
new_coeffs = generate_signal_divisible_by_4(coefficients)

# Reconstruct the new signal from these coefficients
# new_signal = reconstruct_signal(new_coeffs, x)
new_signal = hermval(x, new_coeffs)

F_4s = np.square(np.abs(new_signal))
# plt.plot(F_4s)

new_no = scipy.fft.irfft(new_signal,n=len(no.data))
plt.plot(new_no)

#%%

# F_no = scipy.fft.rfft(no.data.values)
signal = no.data  # Example complex signal
max_order = 100
x = np.arange(len(signal))

# Decompose the signal into Hermite polynomials
# coefficients = decompose_signal_to_hermite(signal, max_order=max_order)
coefficients = hermfit(x,signal, deg=max_order)

# Generate a new signal using only coefficients at indices divisible by 4
new_coeffs = generate_signal_divisible_by_4(coefficients)

# Reconstruct the new signal from these coefficients
# new_signal = reconstruct_signal(new_coeffs, x)
# new_signal = hermval(x, new_coeffs)
new_signal = hermval(x, coefficients)

plt.plot(new_signal)