# Simulate a Stage IV-like weak lensing observation and generate mock galaxy shear catalogues

The catalogue simulation is executed by running ```run_cat_sim.sh``` in bash.

To set up the simulation we need to define the survey parameters in set_variables_cat.ini, and direct the pipeline path
in run_cat_sim.sh. Specifically, in set_variables_cat.ini:

- Change the ```PIPELINE_DIR``` to the root directory where the package is installed, i.e.
```PIPELINE_DIR=/path-in-system/SWEPT/```

- Change the redshift distribution parameters in ```redshift_distribution```. Here ```ZMIN, ZMAX, DZ``` are the minimum redshift, maximum redshift, and resolution to characterise the n(z) redshift distribution for the sample. The parameters ```Z0, BETA``` describe the functional form of the n(z) distribution (see also Eq. 44 in Wong+24). Finally, ```NGAL``` is the number of galaxies to include in the mock sample, expressed in scientific form, i.e. '1e6' for 1m galaxies.

- In ```[create_nz]```, ```NZ_TABLE_NAME``` does not need to be changed. However, we need to specify the number of bins in redshift space to use to generate maps at. Generally, we want the number of bins to match the number of sample points, i.e. ```N_ZBIN=(ZMAX-ZMIN)/DZ```. We recommend that for a full 3x2pt analysis you used ```DZ```=0.1. However,  if the number of sample points exceeds the number of points at which we generate maps to sample the redshift distribution, the code will interpolate field values per pixel, to 'fill in' the missing information in redshift. We recommend that no interpolation should be used for a full 3x2pt cosmological analysis, as the interpolation can introduce correlations that may contaminate precision cosmological constraints. For shear-only, you can get away with using a smaller ```DZ```, but the number of bins may not be able to match the number of sample points (e.g. due to numerical errors if fields in each bin) are generated very close together. In this shear only case, the effects of interpolation will not be as significant as a full 3x2pt analysis (since the shear signal is reasonably well described by a cumulative sum/interpolation along the line of sight)

- In simulation set up, change the ```SIMULATION_SAVE_DIR``` to the path where you want the catalogues to be saved. Also change the ```NSIDE```, ```ELL``` range, and number of ```REALISATIONS``` as required. The mock observation will be generated over a given survey mask. This mask must be saved, in ```HEALPIX``` format on your local machine. The ```PATH_TO_MASK``` is the path to where your survey mask is saved on disk. Note that the ```NSIDE``` of the mask must match the ```NSIDE``` specified in the parameter space otherwise the code will have unexpected behaviour!

- In ```[compile_cat]```, change the errors injected into the redshift photo-z estimates and galaxy/shear shapes.

Change the ```COSMOSIS_ROOT_DIR``` to the directory where ```CosmoSIS``` is installed on disk.

In ```run_cat_sim.sh```, change the ```PIPELINE_VARIABLES_PATH``` to the path where the ```set_variables_cat.ini``` file is located, i.e.
```PIPELINE_VARIABLES_PATH=/path-in-system/SWEPT/catalogue_sim/set_variables_cat.ini```
