import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

np.seterr(divide='ignore')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

save_dir = '/raid/scratch/wongj/mywork/3x2pt/FINAL_DATA/NONOISE/EQUIPOP/3BIN_EQUIPOP/'
nbins = 3
bins = np.arange(1, nbins + 1, 1)


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


ymin_vals = []
ymax_vals = []

for j in bins:
    for i in bins:
        bp = open_dat(save_dir + 'theory_cls/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(i, j))
        ell = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/ell.txt')

        bp_measured = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(i, j))
        bp_err = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}_err.txt'.format(i, j))
        f, ax1 = plt.subplots()
        plt.plot(ell, bp, label='Theoretical Spectra', color='black', zorder=0.75)
        plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colors[3], label='Measured Spectra',
                     linestyle='None', marker='x', markersize=7.5, zorder=10)

        ymin, ymax = ax1.get_ylim()
        ymin_vals.append(ymin)
        ymax_vals.append(ymax)

        plt.close()

ymins = []
ymaxs = []

id_arr = []
val = 0
id_arr.append(val)
for i in range(nbins):
    val += nbins
    id_arr.append(val)

for i in range(len(id_arr)-1):
    ymins.append(ymin_vals[id_arr[i]:id_arr[i+1]])
    ymaxs.append(ymax_vals[id_arr[i]:id_arr[i+1]])


fig = matplotlib.pyplot.figure(figsize=(10, 10))
sz = 1.0 / (nbins + 2)

for i in bins:
    for j in bins:

        bp = open_dat(save_dir + 'theory_cls/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(i, j))
        ell = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/ell.txt')

        bp_measured = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(i, j))
        bp_err = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}_err.txt'.format(i, j))

        rect = (i*sz,j*sz,sz,sz)
        #rect = ((i * sz) + (0.08 * i) - 0.15, (j * sz) + (0.04 * j) - 0.0875, 1.25 * sz, sz)
        ax = fig.add_axes(rect)

        plt.plot(ell, bp, label='Theoretical Spectra', color='black', zorder=0.75)

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        fig.canvas.draw()
        offset = ax.yaxis.get_major_formatter().get_offset()

        plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colors[3], label='Measured Spectra',
                     linestyle='None', marker='x', markersize=7.5, zorder=10)
        plt.xscale('log')

        #if i == 1 and j == 1:
        #    ax.legend(bbox_to_anchor=(0.9, 1.6), fontsize=13.5)

        if j == 1:
            plt.xlabel("$\\ell$", fontsize=15)

        if i == 1:
            labelstr = str("$C_b^{\delta_{g}\gamma}$")
            plt.ylabel(labelstr, fontsize=15)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_visible(False)
            ax.text(0.01, 0.99, offset, ha='left', va='top', transform=ax.transAxes)

        if j != 1:
            plt.gca().xaxis.set_ticklabels([])

        if i != 1:
            plt.gca().yaxis.set_ticklabels([])

        if len(ymins[j-1]) == 1:
            scale_factor = -1.1
        else:
            scale_factor = -10

        ax.set_ylim(scale_factor * abs(min(ymins[j - 1])), 1.2 * max(ymaxs[j - 1]))

        ax.minorticks_on()

        ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                       bottom=True, labelleft=True, labelbottom=True, direction='in')
        ax.tick_params(length=2.5, which='minor')
        ax.tick_params(length=5.5, which='major')
        ax.tick_params(labelsize=12.5)

        plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i, j), fontsize=15, color='black',
                 transform=ax.transAxes)

        # plt.title('NGal = 5e6, 10 Realisations')

plt.savefig(save_dir + 'galaxy_shear_cl.png')
plt.show()


fig = matplotlib.pyplot.figure(figsize=(10, 10))

for i in bins:
    for j in bins:

        bp = open_dat(save_dir + 'theory_cls/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(i, j))
        ell = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/ell.txt')

        bp_measured = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(i, j))
        bp_err = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_shear_bp/bin_{}_{}_err.txt'.format(i, j))

        yerr = (bp_err / bp)
        yerr[yerr == np.Inf] = np.NaN

        # rect = ((i * sz) + (0.08 * i) - 0.15, (j * sz) + (0.04 * j) - 0.0875, 1.25 * sz, sz)
        rect = (i*sz,j*sz,sz,sz)
        ax = fig.add_axes(rect)

        plt.errorbar(ell, ((bp_measured) / (bp)) - 1, xerr=None, yerr=yerr, label='Fractional Difference',
                     linestyle='None', marker='x', markersize=7.5, color=colors[3], zorder=1)

        plt.axhline(y=0, color='black')
        plt.xscale('log')

        #if i == 1 and j == 1:
        #    ax.legend(bbox_to_anchor=(0.85, 1.6), fontsize=13.5)

        if j == 1:
            plt.xlabel("$\\ell$", fontsize=15)

        if i == 1:
            labelstr = str(
                "$\\frac{C_b^{\delta_{g}\gamma, \\mathrm{Measured}}}{C_b^{\delta_{g}\gamma, \\mathrm{Theory}}}-1$")
            plt.ylabel(labelstr, fontsize=15)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        if j != 1:
            plt.gca().xaxis.set_ticklabels([])

        if i != 1:
            plt.gca().yaxis.set_ticklabels([])

        ax.set_ylim([-0.05, 0.05])

        ax.minorticks_on()

        ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                       bottom=True, labelleft=True, labelbottom=True, direction='in')
        ax.tick_params(length=2.5, which='minor')
        ax.tick_params(length=5.5, which='major')
        ax.tick_params(labelsize=12.5)

        plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i, j), fontsize=15, color='black',
                 transform=ax.transAxes)
        # plt.title('NGal = 5e6, 10 Realisations')

plt.savefig(save_dir + 'galaxy_shear_cl_residuals.png')
plt.show()

