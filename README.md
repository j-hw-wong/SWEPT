# Simulator for WEak Lensing Power spectrum Tomography (SWEPT)

This code is set up to do three components of a 3x2pt cosmological analysis that can be executed 
individually or consecutively:

1) The generation of a set of mock galaxy catalogues that simulate the observation of a next gen-like weak lensing 
survey - handled by the ```catalogue_sim/``` package
2) The measurement of the 3x2pt Power Spectra from the mock catalogues for a specified tomographic configuration - 
using the Pseudo-Cl estimator for the signal and analytic expressions for the noise Cls, which are then converted into 
bandpowers - handled by the ```pcl_measurement/``` package
3) The inference analysis to constrain the w0-wa Dark Energy Equation of State parameters from the measured 3x2pt data 
vector - handled by the ```inference_analysis/``` package. These rely on the ```gaussian_cl_likelihood``` and ```angular binning```
packages, which have been forked/redeveloped from https://github.com/robinupham/gaussian_cl_likelihood, https://github.com/robinupham/angular_binning (Upham+21)

Each of these folders come with relevant READMEs in the file structure to give a walkthrough on how to run the relevant
simulation/analysis. Each (and all) of these components of the end-to-end 3x2pt cosmological analysis is defined for a 
fiducial cosmology, which is specified/set in the ```software_utils``` directory - please see the README and config files here
for information on how to set the global fiducial cosmology for the analysis.

A description and demonstration of this code run fully from end to end is also presented in '*Euclid*: Optimising 
tomographic binning for 3x2pt power spectrum constraints on dark energy,' Wong+24 (in review).

This code is designed to run on a laptop/desktop, so is generally serial in nature. Code for parallel use at greater
resolution on HPC cluster is under construction/available on request.

The list of python dependencies for all three components is found in ```REQUIREMENTS.TXT```. In addition to python 
packages, this code requires two additional software installations on the local machine:

- ```CosmoSIS``` (Zuntz+18), https://arxiv.org/pdf/1409.3409, https://bitbucket.org/joezuntz/cosmosis/wiki/Home
- ```FLASK``` (Xavier+16), https://arxiv.org/pdf/1602.08503, http://www.astro.iag.usp.br/~flask/

For any questions, please contact jonathan.wong@manchester.ac.uk or jonathanhw.wong@gmail.com
