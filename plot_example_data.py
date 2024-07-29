# plot and generate results
import os
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from funcs.dependencies_clf import interpret_bayes_factor_log
from PIL import Image

# load data
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'sensitivity')

matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
cms = 1/2.54  # centimeters in inches
freq_vector = np.arange(1, 48)
# Plot individual subject data upper and lower panel of the figure
# 3 controls and 3 super recognizers to illustrate variability
# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
results_dir = os.path.join(main_dir, 'results', 'classification_data')
fig, axs = plt.subplots(2, 3, figsize=(18 * cms, 6 * cms))
fig.subplots_adjust(hspace=0, wspace=0)
# select which experiment you want to plot / change experiment_id
fig.suptitle('Experiment 3', y=0.89, fontstyle='italic')
experiment_id = 3
# results from experiment
results = list(filter(lambda s: '_acc_expt_3.npy' in s, os.listdir(results_dir)))

idx_sr = [5, 9, 13]
idx_con = [4, 3, 11]
all_ix = idx_sr + idx_con
color_ = ['k', 'k', 'k', 'grey', 'grey', 'grey']
ixp = 0
ax_pl = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
delticks = 0
# change idx_sr to idx_con. SRs are 3_2
# change color 'red' to 'blue' for controls
for id_ in all_ix:

    subj = results[id_]
    subj_id = subj.replace('_acc_expt_3.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    group_ = demog_file.loc[subj_id].Label
    # plt.subplot(2, 3, ixp + 1)

    # true data
    prc_ = np.percentile(data_, (25, 50, 75), axis=0)
    prc_ = prc_[:, 1:-1]
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]
    ax1 = ax_pl[ixp]
    axs[ax1[0], ax1[1]].scatter(freq_vector, prc_[1], s=5, c=color_[ixp], alpha=1)
    axs[ax1[0], ax1[1]].errorbar(freq_vector, prc_[1], yerr=prc_[[0, 2]], capsize=1, color=color_[ixp], elinewidth=0.5, linewidth=0.6, alpha=1)

    if experiment_id == 1:
        axs[ax1[0], ax1[1]].set_xlim(0.5, 47.5)
        axs[ax1[0], ax1[1]].set_ylim(0.18, 0.68)  # Adjust according to your data range
        axs[ax1[0], ax1[1]].plot([-1, 100], [0.25, 0.25], linestyle='--', color='green', alpha=0.4)
        axs[ax1[0], ax1[1]].set_yticks([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
        axs[ax1[0], ax1[1]].set_xticks([10, 20, 30, 40], [10, 20, 30, 40])

    elif experiment_id == 2:
        axs[ax1[0], ax1[1]].set_xlim(0.5, 47.5)
        axs[ax1[0], ax1[1]].set_ylim(0.44, 0.78)  # Adjust according to your data range
        axs[ax1[0], ax1[1]].plot([-1, 100], [0.5, 0.5], linestyle='--', color='green', alpha=0.4)
        axs[ax1[0], ax1[1]].set_yticks([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        axs[ax1[0], ax1[1]].set_xticks([10, 20, 30, 40], [10, 20, 30, 40])

    else:
        axs[ax1[0], ax1[1]].set_xlim(0.5, 47.5)
        axs[ax1[0], ax1[1]].set_ylim(0.28, 0.79)  # Adjust according to your data range
        axs[ax1[0], ax1[1]].plot([-1, 100], [0.3333, 0.3333], linestyle='--', color='green', alpha=0.4)
        axs[ax1[0], ax1[1]].set_yticks([0.3, 0.5, 0.7], [0.3, 0.5, 0.7])
        axs[ax1[0], ax1[1]].set_xticks([10, 20, 30, 40], [10, 20, 30, 40])

    if ixp == 4:
        axs[ax1[0], ax1[1]].set_xlabel('frequency')

    ixp = ixp + 1


# Add shared y-axis label
fig.text(0, 0.5, 'accuracy', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3_ex.jpg'), dpi=500)


# Create final figure by stacking the generated subplots
image1 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_1_ex.jpg'))
image2 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_2_ex.jpg'))
image3 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3_ex.jpg'))

# Get dimensions of the images
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size

# Calculate the cropping dimensions
crop_height = int(height1 * 0.01)  # 10% of the height of image1

# Crop 10% of the lower part of image1
image1_cropped = image1.crop((0, 0, width1, height1 - crop_height))

# Crop 10% of the lower part of image3
image3_cropped = image3.crop((0, 0, width3, height3 - crop_height))

# Create a new blank image with combined dimensions
new_width = max(width1, width2, width3)
new_height = height1 - crop_height + height2 + height3 - crop_height
combined_image = Image.new("RGB", (new_width, new_height))

# Paste the cropped image1 onto the new image at (0,0) position
combined_image.paste(image1_cropped, (0, 0))

# Paste the second image below the cropped image1
combined_image.paste(image2, (0, height1 - crop_height))

# Paste the cropped image3 below image2
combined_image.paste(image3_cropped, (0, height1 + height2 - 2 * crop_height))

# Save or display the combined image
combined_image.save(os.path.join(main_dir, 'results', 'figures', 'example_data.pdf'))
combined_image.show()
