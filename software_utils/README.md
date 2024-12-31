# Here we will set parameters to define the global cosmology for the ```SWEPT``` analysis.

The global cosmology is used to generate the mock catalogue data, measure the Pseudo-Cl power spectra, and perform the inference analysis to derive
constraints on w0-wa. (For the likelihood analysis for the latter, all cosmological parameters are fixed except for w0, wa). For each of the files here:

- ```cosmosis_config``` - the config file used to specify which routines to run in ```CosmoSIS```. For the setup presented in Wong+24, this does not need to be changed
- ```cosmosis_params``` - used to specify the cosmological parameters (e.g. $\Omega_{\mathrm{m}}, n_{s}, A_{s}$) in CosmoSIS. These parameters are as specified in Sect.4 in Wong+24 but can be changed to an arbitrary cosmology
- ```flask_3x2pt.config``` - a template of parameters used to run ```FLASK```. Note that this is just a template that is necessary for ```FLASK``` to be run - the parameters are overwritten in ```catalogue_sim.run_flask``` based on the ```catalogue_sim/set_variables_cat.ini``` config file. So the file here should not ever need to be changed! To run ```FLASK``` with different parameters (e.g. $\ell$ ranges, this can be controlled in the ```catalogue_sim/set_variables_cat.ini``` config file 
- ```run_cosmosis.sh``` - a shell script to run ```CosmoSIS``` based on the routines specified in ```cosmosis_config```
