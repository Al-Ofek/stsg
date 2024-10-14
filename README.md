# STSG

This repository contains the source code and data used in "Synthetic Random Environmental Time Series Generation with Similarity Control, Preserving Original Signal's Statistical Characteristics" by Ofek Aloni, Gal Perelman, and Barak Fishbain.

## Structure
stsg.py is the main file, containing the STSG class. It offers methods for importing data, generating synthetic time series using the Fourier method as in the paper, and methods used for calculating statistics and metrics of the generated series.

A sample workflow can be found in any of the "{data name}_workfile.py" files, that are identical in structure but contain slight modifications required by each of the different data sets. The data is imported, then each method in turn is used for generation of 1,000 synthetic samples. The statistics and metrics are calculated for each, and finally everything is summarized in a df. In addition, a plot with a sample time series from each method is plotted, along with a histogram and autocorrelation function.

The Fourier method and ARMA are built into to the STSG class, but the other methods require some additional notes:

### QuantGAN
QuantGAN model was implemented using Pytorch. Since training a GAN is computationally intensive, a virtual machine was used for training. The file used for training is included, as well as a variety of generator models pre-trained on the used datasets, under "\GAN\{data name} Generator". For this reason the workfiles only import a model, then use it for generating synthetic series.

### CoSMoS
Since CoSMoS is available only as an R package during the time of writing, the synthetic data for it was produced seperately, then imported into Python. Working files and data can be found in the cosmos directory. For more on the packge, see https://CRAN.R-project.org/package=CoSMoS.



