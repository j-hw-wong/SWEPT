Pseudo-Cl measurement
===================================

This module measures the tomographic 3x2pt Pseudo-Cl power spectra from mock shear catalogues.

Following the creation of mock shear catalogues using ``run_cat_sim.sh`` in ``SWEPT.catalogue_sim``, we can measure the 3x2pt Pseudo-Cl power spectra from
the mock catallogues for a given tomographic binning configuration using ``run_3x2pt_tomo_measurement.sh`` in this ``pcl_measurement`` package.

Explicitly, we will first need to set the parameters of the measurement in the ``set_ variables_3x2pt_measurement.ini`` config file before executing code for power
spectrum measurement. A description of these parameters to specify is included in the header/comments of ``set_ variables_3x2pt_measurement.ini``.

Before running the Pseudo-Cl measurement, we then need to set the path to the parameter config file:

- In ``run_3x2pt_tomo_measurement.sh``, set ``PIPELINE_VARIABLES_PATH`` to the location of the config file on disk, i.e. ``PIPELINE_VARIABLES_PATH="/local-path-to/SWEPT/pcl_measurement/set_variables_3x2pt_measurement.ini"``

Then to execute the tomographic Pseudo-Cl measurement, run ``run_3x2pt_tomo_measurement.sh`` in bash. The code will measure the Pseudo-Cl power spectra, binned into
angular bandpowers, from the simulated data/mock shear catalogues, and save the measured spectra on disk. Based on the simulation settings and fiducial cosmology (as specified in ``set_ variables_3x2pt_measurement.ini``), the code will also generate a predicted/theoretical signal to fit to the measured simulated data. The predicted signal and measured signal will be saved in txt format in a location specified in the ``set_ variables_3x2pt_measurement.ini`` config file. Finally, the code will also create a numerical covariance matrix from the measured 3x2pt PCl data, which can be used for cosmological inference analyses (e.g in ``SWEPT/inference_analysis/``)


.. toctree::
   :maxdepth: 4

   pcl_measurement/av_cls
   pcl_measurement/conv_bps
   pcl_measurement/cov_fromsim
   pcl_measurement/create_nz_boundaries
   pcl_measurement/measure_cat_3x2pt_pcls
   pcl_measurement/measure_cat_bps
   pcl_measurement/run_3x2pt_tomo_measurement.md
   pcl_measurement/set_variables_3x2pt_measurement.md

