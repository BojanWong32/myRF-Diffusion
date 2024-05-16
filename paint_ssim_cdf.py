import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import scipy.io as scio
import os

data_root = './'
save_root = './'

import matplotlib


overall_rot_file = os.path.join(data_root, 'exp_overall_ssim_fmcw.mat')
sigma = scio.loadmat(overall_rot_file)['data_fmcw_sigma']
sigma_std = np.std(sigma)
sigma_mean = np.mean(sigma)

w_perc = np.percentile(sigma, 90)

n_bins = np.arange(0, 1, 0.0001)
plt.figure(figsize=(4, 2.5))
ax = plt.subplot()

# Data
counts_1, _ = np.histogram(sigma, bins=n_bins, density=True)
cdf_1 = np.cumsum(counts_1)
cdf_1 = cdf_1.astype(float) / cdf_1[-1]




# seagreen darkorange indianred steelblue
blue = '#084E87'
orange = '#ef8a00'
green = '#267226'
red = '#BF3F3F'

plt.plot(n_bins[0:-1], cdf_1, '-', zorder=4,color=blue,linewidth=2, label='RF-Diffusion')

# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    # label.set_fontproperties(font)
    label.set_fontsize(11)
plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.ylim(0, 1)
plt.xlim(0, 1.0)
plt.xlabel('SSIM', verticalalignment='top')
plt.ylabel('CDF', verticalalignment='bottom')
leg = plt.legend(loc='best', prop={'size': 9})
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
plt.tight_layout()
plt.show()
# plt.savefig(save_root + '/Fig7(a)-exp-overall-fmcw-ssim.pdf', dpi=800)