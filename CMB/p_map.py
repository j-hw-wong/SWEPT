import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cl_dir = '/raid/scratch/wongj/mywork/XCORR/Test_Fields/Simulation/cosmosis/cmb_cl/'

ell = np.loadtxt(cl_dir + 'ell.txt')

bb = np.loadtxt(cl_dir + 'bb.txt')
ee = np.loadtxt(cl_dir + 'ee.txt')
te = np.loadtxt(cl_dir + 'te.txt')
tt = np.loadtxt(cl_dir + 'tt.txt')

pp = np.loadtxt(cl_dir + 'pp.txt')

map_pp = hp.synfast(pp, nside=128, pol=False)
hp.mollview(map_pp)
plt.show()
plt.close()
