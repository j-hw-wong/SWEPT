import numpy as np
import matplotlib.pyplot as plt

cl_dir = '/raid/scratch/wongj/mywork/XCORR/Test_Fields/Simulation/cosmosis/cmb_cl/'

ell = np.loadtxt(cl_dir + 'ell.txt')

bb = np.loadtxt(cl_dir + 'bb.txt')
ee = np.loadtxt(cl_dir + 'ee.txt')
te = np.loadtxt(cl_dir + 'te.txt')
tt = np.loadtxt(cl_dir + 'tt.txt')

pp = np.loadtxt(cl_dir + 'pp.txt')

pi = np.pi

def conv_cl(ell, cl):
    return ((ell*ell+1)*cl)/(2*pi)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12.8, 4.8))

ax1.plot(ell, bb, label='bb')
ax1.plot(ell, ee, label='ee')
ax1.plot(ell, abs(te), label='te')
ax1.plot(ell, tt, label='tt')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()

ax2.plot(ell, pp, label='pp')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()

plt.show()
