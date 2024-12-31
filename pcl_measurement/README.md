# Measure tomographic 3x2pt Pseudo-Cl power spectra from mock shear catalogues

Following the creation of mock shear catalogues using ```run_cat_sim.sh``` in ```SWEPT.catalogue_sim```, we can measure the 3x2pt Pseudo-Cl power spectra from
the mock catallogues for a given tomographic binning configuration using ```run_3x2pt_tomo_measurement.sh``` in this ```pcl_measurement``` package.

Explicitly, we will first need to set the parameters of the measurement in the ```set_ variables_3x2pt_measurement.ini``` config file before executing code for power
spectrum measurement. A description of these parameters to specify is included in the header/comments of ```set_ variables_3x2pt_measurement.ini```.

Before running the Pseudo-Cl measurement, we then need to set the path to the parameter config file:

- In ```run_3x2pt_tomo_measurement.sh```, set ```PIPELINE_VARIABLES_PATH``` to the location of the config file on disk, i.e. ```PIPELINE_VARIABLES_PATH="/local-path-to/SWEPT/pcl_measurement/set_variables_3x2pt_measurement.ini"```

Then to execute the tomographic Pseudo-Cl measurement, run ```run_3x2pt_tomo_measurement.sh``` in bash.
