# Simulator for WEak Lensing Power spectrum Tomography

This code is set up to do three components of a 3x2pt cosmological analysis that can be executed 
individually or consecutively:

1) The generation of a set of mock galaxy catalogues that simulate the observation of a next gen-like weak lensing 
survey - handled by the catalogue_sim/ package
2) The measurement of the 3x2pt Power Spectra from the mock catalogues for a specified tomographic configuration - 
using the Pseudo-Cl estimator for the signal and analytic expressions for the noise Cls, which are then converted into 
bandpowers - handled by the pcl_measurement/ package
3) The inference analysis to constrain the w0-wa Dark Energy Equation of State parameters from the measured 3x2pt data 
vector - handled by the inference_analysis/ package

Each of these folders come with relevant READMEs in the file structure to give a walkthrough on how to run the
simulation/analysis.

A description and demonstration of this code run fully from end to end is also presented in '_Euclid:_Optimising 
tomographic binning for 3x2pt power spectrum constraints on dark energy,' Wong+24 (in review).

This code is designed to run on a laptop/desktop, so is generally serial in nature. Code for parallel use at greater
resolution on HPC cluster is under construction/available on request.

Please contact jonathan.wong@manchester.ac.uk
