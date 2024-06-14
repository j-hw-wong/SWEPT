The catalogue simulation is executed by running 'run_cat_sim.sh' in bash.

To set up the simulation we need to define the survey parameters in set_variables_cat.ini, and direct the pipeline path
in run_cat_sim.sh. Specifically:

In set_variables_cat.ini:

Change the PIPELINE_DIR to the root directory where the package is installed, i.e.
PIPELINE_DIR=/path-in-system/Jonathans_Big_Cosmology_Automator/

Change the redshift distribution parameters, including the number of galaxies.

In [create_nz], NZ_TABLE_NAME does not need to be changed. However, we need to specify the number of bins in redshift
space. This is essentially how many bins you want to sample the redshift distribution in. For no interpolation, the
number of bins must match the number of sample points, i.e. N_ZBIN=(ZMAX-ZMIN)/DZ. We recommend that for a full 3x2pt
analysis you used DZ=0.1 and no interpolation. For shear-only, you can get away with using a smaller DZ, but the number
of bins may not be able to match the number of sample points (e.g. due to numerical errors if fields in each bin) are
generated very close together. In this case, the simulation will attempt to interpolate field values for each pixel
between bins/

In simulation set up, change the SIMULATION_SAVE_DIR to the path where you want the catalogues to be saved. Also change
the NSIDE, ELL range, and number of REALISATIONS. The PATH_TO_MASK is the path to where your survey mask is saved on
disk. Note that the NSIDE of the mask must match the NSIDE specified in the parameter space otherwise the code will
have unexpected behaviour!

In [compile_cat], change the errors injected into the Photo_zs and shear shapes.

Change the COSMOSIS_ROOT_DIR to the directory where CosmoSIS is installed on disk.

In run_cat_sim.sh, change the PIPELINE_VARIABLES_PATH to the path where the set_variables_cat.ini file is located, i.e.
PIPELINE_VARIABLES_PATH=/path-in-system/Jonathans_Big_Cosmology_Automator/catalogue_sim/set_variables_cat.ini
