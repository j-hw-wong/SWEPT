# Perform likelihood analysis to derive Dark Energy constraints from (simulated) 3x2pt tomographic Pseudo-Cl (PCl) power spectra

Following the creation of mock shear catalogues using ```run_cat_sim.sh``` in ```SWEPT/catalogue_sim```, and the measurement of tomographic 3x2pt PCl spectra from
the simulated mock data using ```SWEPT/pcl_measurement/run_3x2pt_tomo_measurement.sh```, we can finally use the measured PCl data to place constraints on Dark Energy
using this ```inference_analysis``` package.

Explicitly, run ```run_inference.sh``` in bash, which will perform a grid-based Gaussian likelihood analysis (presented in Upham+21, https://github.com/robinupham/gaussian_cl_likelihood)
to derive constraints on the w0-wa parameters for a time-evolving Dark Energy equation of state. The parameters/set up of the inference analysis is controlled by the
```set_variables_inference.ini``` config file and the ```setup_inference_grid.py``` script. So, before executing ```run_inference.sh``` we will need to:

- Set the path to the config file: in ```run_inference.sh```, set ```PIPELINE_VARIABLES_PATH="/path-on-disk-to/SWEPT/inference_analysis/set_variables_inference.ini"```
- Specify inference analysis parameters in ```set_variables_inference.ini```. Detailed description of parameters is provided in the header/comments in this config file
- In ```setup_inference_grid.py```, set the ranges/priors on the w0-wa grid to perform the likelihood analysis

By running ```run_inference.sh```, the code will perform the Gaussian likelihood analysis, and save the posterior constraints on w0-wa for the given dataset/tomographic
configuration in txt format on disk (location specified in the ```set_variables_inference.ini``` config file.

[The implementation of the Hartlap Correction will need to be input manually within ```angular_binning```
