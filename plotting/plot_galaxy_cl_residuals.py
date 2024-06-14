import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

np.seterr(divide='ignore')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

colour = 'royalblue'

save_dir = '/raid/scratch/wongj/mywork/3x2pt/FINAL_DATA/NOISE/EQUIPOP/3BIN_EQUIPOP/'
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
        if i >= j:
            bp = open_dat(save_dir + 'theory_cls/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(i, j))
            ell = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/ell.txt')

            bp_measured = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/bin_{}_{}.txt'.format(i, j))
            bp_err = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/bin_{}_{}_err.txt'.format(i, j))
            f, ax1 = plt.subplots()
            plt.plot(ell, bp, label='Theoretical\nSpectra', color='black', zorder=0.75)
            plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colour, label='Measured\nSpectra',
                         linestyle='None', marker='.', markersize=7.5, zorder=10)

            ymin, ymax = ax1.get_ylim()
            ymin_vals.append(ymin)
            ymax_vals.append(ymax)

            plt.close()

ymins = []
ymaxs = []

id_arr = []
val = 0
id_arr.append(val)
for i in reversed(bins):
    val += i
    id_arr.append(val)

for i in range(len(id_arr)-1):
    ymins.append(ymin_vals[id_arr[i]:id_arr[i+1]])
    ymaxs.append(ymax_vals[id_arr[i]:id_arr[i+1]])


fig = matplotlib.pyplot.figure(figsize=(10, 10))
sz = 1.0 / (nbins + 2)

#xtick_arr = np.arange(200, 1400, 200)
xtick_arr = np.array([250, 500, 1000])
#xtick_arr = np.logspace(np.log10(100), np.log10(1500), 11)
#print(xtick_arr)
for j in bins:
    for i in bins:
        if i >= j:
            bp = open_dat(save_dir + 'theory_cls/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(i, j))
            ell = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/ell.txt')

            bp_measured = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/bin_{}_{}.txt'.format(i, j))
            bp_err = open_dat(save_dir + 'measured_3x2pt_bps/galaxy_bp/bin_{}_{}_err.txt'.format(i, j))

            #rect = (i*sz,j*sz,sz,sz)
            rect = ((i * sz) + (0.08 * i) - 0.15, (j * sz) + (0.1 * j) - 0.145, 1.25 * sz, sz)
            ax = fig.add_axes(rect)
            plt.axhline(y=0, color='0.5',lw=1.25,ls=':')

            plt.plot(ell, bp, label='Theoretical\nSpectra', color='black', zorder=0.75,lw=1.25)
            plt.errorbar(ell, bp_measured, xerr=None, yerr=bp_err, color=colour, label='Measured\nSpectra',
                         linestyle='None', marker='x', markersize=7, zorder=10)
            plt.xscale('log')

            #if i == 1 and j == 1:
                #ax.legend(bbox_to_anchor=(0.65, 1.9), fontsize=13.5)

            if i == j:
                labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                plt.ylabel(labelstr, fontsize=20)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            plt.gca().xaxis.set_ticklabels([])

            if i != j:
                plt.gca().yaxis.set_ticklabels([])

            if len(ymins[j-1]) == 1:
                scale_factor = -5
            elif len(ymins[j-1]) == 2:
                scale_factor = -10
            else:
                scale_factor = -500

            ax.set_ylim(scale_factor*abs(min(ymins[j-1])), 1.2*max(ymaxs[j-1]))
            ax.set_xticks(xtick_arr)
            #ax.set_xticklabels(xtick_arr)
            #ax.minorticks_on()

            ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                           bottom=True, labelleft=True, labelbottom=True, direction='in')
            ax.tick_params(length=2.5, which='minor')
            ax.tick_params(length=5.5, which='major')
            ax.tick_params(labelsize=15)
            ax.yaxis.get_offset_text().set_fontsize(15)
            plt.xticks(fontsize=17.5)
            plt.yticks(fontsize=17.5)

            plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i, j), fontsize=17.5, color='black',
                     transform=ax.transAxes)

            yerr = (bp_err / bp)
            yerr[yerr == np.Inf] = np.NaN

            rect2 = ((i * sz) + (0.08 * i) - 0.15, (j * sz) + (0.1 * j) - 0.22, 1.25 * sz, sz*0.375)
            #rect2 = ((i * sz) + (0.08 * i) - 0.15, (j * sz) + (0.04 * j) - 0.275, 1.25 * sz, sz*0.5)
            ax2 = fig.add_axes(rect2)

            residual = ((bp_measured) / (bp)) - 1

            residual[residual == np.Inf] = 0
            residual[residual == -np.Inf] = 0

            if i == j:
                #labelstr = str("$\\frac{C_\ell^{\delta_{g}\delta_{g}, \\mathrm{M}}}{C_b^{\delta_{g}\delta_{g}, \\mathrm{T}}}-1$")
                labelstr = str("$\Delta_{f}$")
                plt.ylabel(labelstr, fontsize=20)
                #plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            if i == 3 and j == 1:

                ax2.errorbar(ell, residual, xerr=None, yerr=None, label='Fractional\nDifference',
                             linestyle='None', marker='None', markersize=0, color=colour, zorder=1, capsize=3)

                for ell_mode in ell:
                    ax2.annotate(s='', xy=(ell_mode, -0.075), xytext=(ell_mode, 0.075),
                                 arrowprops=dict(arrowstyle='<->', color=colour))


            else:
                ax2.errorbar(ell, residual, xerr=None, yerr=yerr, label='Fractional\nDifference',
                             linestyle='None', marker='x', markersize=7, color=colour, zorder=1, capsize=3)


            plt.axhline(y=0, color='black',lw=1.25)
            plt.xscale('log')

            ax2.set_xticks(xtick_arr)
            #ax2.tick_params('x', length=0, which='major')

            if j == 1:
                plt.xlabel("$\\ell$", fontsize=20)
                ax2.set_xticklabels(xtick_arr)

            if j != 1:

                plt.gca().xaxis.set_ticklabels([])

            if i!=j:
                plt.gca().yaxis.set_ticklabels([])

            ax2.set_ylim([-0.15, 0.15])

            #ax2.minorticks_on()

            ax2.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                           bottom=True, labelleft=True, labelbottom=True, direction='in')
            ax2.tick_params(length=2.5, which='minor')
            ax2.tick_params(length=5.5, which='major')
            ax2.tick_params(labelsize=15)

            plt.xticks(fontsize=17.5)
            plt.yticks(fontsize=17.5)


plt.savefig(save_dir + 'galaxy_cl.png')
plt.show()

