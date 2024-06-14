"""
Script to delete catalogue simulation by-products that have been saved on disk. Important for considerations of disk
space, especially for large numbers of galaxies and high redshfit ranges. Deletes the interpolated field maps and
the catalogue indices by-product following Poisson sampling. Repeated over a given number of realisations/iterations.
"""

import os
import shutil
import configparser

pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

config = configparser.ConfigParser()
config.read(pipeline_variables_path)

iter_no = int(float(os.environ['ITER_NO']))

save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
interp_maps_dir = save_dir + 'flask/interp_maps/'

fields = ['clustering/', 'convergence/', 'shear/']

for field in fields:
    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(interp_maps_dir+field+'iter_{}/'.format(iter_no))
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    with open(interp_maps_dir+field+'README.txt', 'a+') as f:
        f.write('\nIter {}: Interp Maps Deleted and Cleaned'.format(iter_no))

cat_indices_dir = save_dir + 'cat_products/cat_indices/'

try:
    shutil.rmtree(cat_indices_dir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

with open(save_dir + 'cat_products/README.txt', 'a+') as f:
    f.write('\n Iter {}: Cat Indices Deleted and Cleaned'.format(iter_no))
