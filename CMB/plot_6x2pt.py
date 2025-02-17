import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from collections import defaultdict

simulation_save_dir = '/raid/scratch/wongj/mywork/XCORR/Fields_XCORR/Simulation/'

no_realisations = 50
nbins = 5
lmax = 250

def open_dat(fname):
    dat_arr = []
    with open(fname) as f:
        for line in f:
            column = line.split()
            if not line.startswith('#'):
                dat_i = float(column[0])
                dat_arr.append(dat_i)
    dat_arr = np.asarray(dat_arr)
    return dat_arr


def process_cls(save_dir, no_iters, type, bin_i=None, bin_j=None):

    if type not in ['kk', 'ky', 'kd', 'yy', 'dd', 'dy']:
        print('Warning! XCorr Type Not Recognised - Exiting...')
        sys.exit()

    cls = []
    measured_cls = []

    for n in range(no_iters):

        if type == 'kk':
            if n == 0:
                cls.append("$C_\ell^{\kappa_{\mathrm{CMB}}\kappa_{\mathrm{CMB}}}$")
                cls.append(open_dat(save_dir + 'cosmosis/cmbkappa_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/cmbkappa_cl/bin_1_1.txt'))

            field_map = hp.read_map(save_dir + 'flask/output/iter_{}/map-f1z{}.fits'.format(n + 1, nbins + 1), field=0)
            cl_measured = hp.anafast(field_map, pol=False, lmax=lmax)
            measured_cls.append(cl_measured[2:])

        elif type == 'ky':
            if n == 0:
                cls.append("$C_\ell^{\kappa_{\mathrm{CMB}}\gamma}$")
                cls.append(open_dat(save_dir + 'cosmosis/shear_cmbkappa_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/shear_cmbkappa_cl/bin_{}_1.txt'.format(bin_i)))

            field_map1 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f1z{}.fits'.format(n + 1, nbins + 1), field=0)
            field_map2 = hp.read_map(
                save_dir + 'flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(n+1, bin_i), field=(0,1,2))

            cl_measured = hp.anafast([field_map1, field_map1, field_map1], field_map2, pol=True, lmax=lmax)[3]
            measured_cls.append(cl_measured[2:])

        elif type == 'kd':
            if n == 0:
                cls.append("$C_\ell^{\kappa_{\mathrm{CMB}}\delta_{\mathrm{g}}}$")
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_cmbkappa_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_cmbkappa_cl/bin_{}_1.txt'.format(bin_i)))

            field_map1 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f1z{}.fits'.format(n + 1, nbins + 1), field=0)
            field_map2 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f2z{}.fits'.format(n+1, bin_i), field=0)

            cl_measured = hp.anafast(field_map1, field_map2, pol=False, lmax=lmax)
            measured_cls.append(cl_measured[2:])

        elif type == 'yy':
            if n == 0:
                cls.append("$C_\ell^{\gamma\gamma}$")
                cls.append(open_dat(save_dir + 'cosmosis/shear_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

            field_map1 = hp.read_map(
                save_dir + 'flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(n + 1, bin_i), field=(0,1,2))
            field_map2 = hp.read_map(
                save_dir + 'flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(n + 1, bin_j), field=(0,1,2))

            cl_measured = hp.anafast(field_map1, field_map2, pol=True, lmax=lmax)[1]
            measured_cls.append(cl_measured[2:])

            # print('Done {} / {}'.format(n+1, no_iters))
        elif type == 'dd':
            if n == 0:
                cls.append("$C_\ell^{\delta_{\mathrm{g}}\delta_{\mathrm{g}}}$")
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

            field_map1 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f2z{}.fits'.format(n + 1, bin_i), field=0)
            field_map2 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f2z{}.fits'.format(n + 1, bin_j), field=0)

            cl_measured = hp.anafast(field_map1, field_map2, pol=False, lmax=lmax)
            measured_cls.append(cl_measured[2:])

        elif type == 'dy':
            if n == 0:
                cls.append("$C_\ell^{\delta_{\mathrm{g}}\gamma}$")
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_shear_cl/ell.txt'))
                cls.append(open_dat(save_dir + 'cosmosis/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

            field_map1 = hp.read_map(
                save_dir + 'flask/output/iter_{}/map-f2z{}.fits'.format(n + 1, bin_i), field=0)
            field_map2 = hp.read_map(
                save_dir + 'flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(n + 1, bin_j), field=(0,1,2))

            cl_measured = hp.anafast([field_map1, field_map1, field_map1], field_map2, pol=True, lmax=lmax)[3]
            measured_cls.append(cl_measured[2:])

    cls.append(np.arange(2, lmax+1))    # ell measured
    av_cls = np.mean(np.array(measured_cls), axis=0)
    cls.append(av_cls)
    return cls

#
# cmb_kks = []
# cmb_kk_theory = open_dat(simulation_save_dir + 'cosmosis/cmbkappa_cl/bin_1_1.txt')
# ell_theory = open_dat(simulation_save_dir + 'cosmosis/cmbkappa_cl/ell.txt')
#
#
# for i in range(no_realisations):
#     cmb_kk_map = hp.read_map(simulation_save_dir + 'flask/output/iter_{}/map-f1z{}.fits'.format(i+1, nbins+1))
#     cmb_kk_measured = hp.anafast(cmb_kk_map, lmax=lmax)
#     cmb_kks.append(cmb_kk_measured)
#
# cmb_kk_measured_av = np.mean(np.array(cmb_kks), axis=0)
# ell_measured = np.arange(0, lmax+1)

cmb_kk_data = process_cls(save_dir=simulation_save_dir, no_iters=50, type='kk')

'''
# CMB convergence
f, a = plt.subplots()
a.plot(cmb_kk_data[1], cmb_kk_data[2])
a.plot(cmb_kk_data[3], cmb_kk_data[4])
a.set_xscale('log')
a.set_yscale('log')
a.set_xlabel("$\\ell$", fontsize=15)
a.set_ylabel(cmb_kk_data[0], fontsize=15)
plt.tight_layout()
plt.show()
'''

def plot_tom_xcorr(save_dir, nbins, no_iters, type, save_name):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for j in range(nbins):
        for i in range(nbins):
            if i >= j:
                # print(i+1, j+1)
                cl_data = process_cls(save_dir=save_dir, no_iters=no_iters, type=type, bin_i=i+1, bin_j=j+1)
                labelstr=cl_data[0]
                ell_theory = cl_data[1]
                cl_theory = cl_data[2]
                ell_measured=cl_data[3]
                cl_measured=cl_data[4]

                rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(ell_theory, cl_theory,zorder=1)
                plt.plot(ell_measured, cl_measured,zorder=2,alpha=0.5)
                plt.xscale('log')
                plt.yscale('log')

                '''
                plt.plot(ell, bp, label='Theoretical\nSpectra', color='black', zorder=0.75)
                plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colors[3], label='Measured\nSpectra',
                             linestyle='None', marker='x', markersize=7.5, zorder=10)

                if i == 1 and j == 1:
                    ax.legend(bbox_to_anchor=(0.65, 1.9), fontsize=13.5)
                '''
                if j == 0:
                    plt.xlabel("$\\ell$", fontsize=15)

                if i == j:
                    # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                    plt.ylabel(labelstr, fontsize=15)
                    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                if j != 0:
                    plt.gca().xaxis.set_ticklabels([])

                if i != j:
                    plt.gca().yaxis.set_ticklabels([])

                # if len(ymins[j-1]) == 1:
                #     scale_factor = -1.1
                # else:
                #     scale_factor = -10
                #
                # ax.set_ylim(scale_factor*abs(min(ymins[j-1])), 1.2*max(ymaxs[j-1]))

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=2.5, which='minor')
                ax.tick_params(length=5.5, which='major')
                ax.tick_params(labelsize=12.5)

                plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i+1, j+1), fontsize=15, color='black',
                         transform=ax.transAxes)
                # plt.title('NGal = 5e6, 10 Realisations')

    plt.savefig(save_name)
    plt.show()



def plot_tom_xcorr2(save_dir, nbins, no_iters, type, save_name):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for j in range(nbins):
        for i in range(nbins):
            if i >= 0:
                # print(i+1, j+1)
                cl_data = process_cls(save_dir=save_dir, no_iters=no_iters, type=type, bin_i=i+1, bin_j=j+1)
                labelstr=cl_data[0]
                ell_theory = cl_data[1]
                cl_theory = cl_data[2]
                ell_measured=cl_data[3]
                cl_measured=cl_data[4]

                rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(ell_theory, cl_theory,zorder=1)
                plt.plot(ell_measured, cl_measured,zorder=2,alpha=0.5)
                plt.xscale('log')
                plt.yscale('log')

                '''
                plt.plot(ell, bp, label='Theoretical\nSpectra', color='black', zorder=0.75)
                plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colors[3], label='Measured\nSpectra',
                             linestyle='None', marker='x', markersize=7.5, zorder=10)

                if i == 1 and j == 1:
                    ax.legend(bbox_to_anchor=(0.65, 1.9), fontsize=13.5)
                '''
                if j == 0:
                    plt.xlabel("$\\ell$", fontsize=15)

                if j == 0:
                    # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                    plt.ylabel(labelstr, fontsize=15)
                    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                if j != 0:
                    plt.gca().xaxis.set_ticklabels([])

                if i != j:
                    plt.gca().yaxis.set_ticklabels([])

                # if len(ymins[j-1]) == 1:
                #     scale_factor = -1.1
                # else:
                #     scale_factor = -10
                #
                # ax.set_ylim(scale_factor*abs(min(ymins[j-1])), 1.2*max(ymaxs[j-1]))

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=2.5, which='minor')
                ax.tick_params(length=5.5, which='major')
                ax.tick_params(labelsize=12.5)

                plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i+1, j+1), fontsize=15, color='black',
                         transform=ax.transAxes)
                # plt.title('NGal = 5e6, 10 Realisations')

    plt.savefig(save_name)
    plt.show()


def plot_tom_xcorr3(save_dir, nbins, no_iters, type, save_name):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for j in range(nbins):
        for i in range(nbins):
            if i == j:
                # print(i+1, j+1)
                cl_data = process_cls(save_dir=save_dir, no_iters=no_iters, type=type, bin_i=i+1, bin_j=j+1)
                labelstr=cl_data[0]
                ell_theory = cl_data[1]
                cl_theory = cl_data[2]
                ell_measured=cl_data[3]
                cl_measured=cl_data[4]

                rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(ell_theory, cl_theory,zorder=1)
                plt.plot(ell_measured, cl_measured,zorder=2,alpha=0.5)
                plt.xscale('log')
                plt.yscale('log')

                '''
                plt.plot(ell, bp, label='Theoretical\nSpectra', color='black', zorder=0.75)
                plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colors[3], label='Measured\nSpectra',
                             linestyle='None', marker='x', markersize=7.5, zorder=10)

                if i == 1 and j == 1:
                    ax.legend(bbox_to_anchor=(0.65, 1.9), fontsize=13.5)
                '''
                plt.xlabel("$\\ell$", fontsize=15)
                plt.ylabel(labelstr, fontsize=15)

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=2.5, which='minor')
                ax.tick_params(length=5.5, which='major')
                ax.tick_params(labelsize=12.5)

                plt.text(0.125, 0.75, "("r'$z_{\mathrm{CMB}}$' ", "r'$z_{%d}$'")" % (j+1), fontsize=15, color='black',
                         transform=ax.transAxes)
                # plt.title('NGal = 5e6, 10 Realisations')

    plt.savefig(save_name)
    plt.show()

plot_tom_xcorr3(save_dir=simulation_save_dir,nbins=nbins, no_iters=no_realisations, type='kd', save_name=simulation_save_dir+'kd.png')
