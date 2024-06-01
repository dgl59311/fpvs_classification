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

matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
cms = 1/2.54  # centimeters in inches

# Generate a diverging colormap
# Define categories and corresponding colors from the colormap
cmap = plt.get_cmap('coolwarm')
categories = ["Decisive (null)", "Strong 2 (null)", "Strong (null)", "Substantial (null)", "Barely worth mentioning",
              "Substantial", "Strong", "Strong 2", "Decisive"]
category_colors = {category: cmap(i / (len(categories) - 1)) for i, category in enumerate(categories)}

# load data
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'sensitivity')

# Experiment
exp_acc = pd.read_csv(os.path.join(results_dir, 'Experiment_3_accuracy.csv'), index_col=0)
controldata = exp_acc[exp_acc['group'] == 'control']
controldata = controldata.drop(['group', 'condition'], axis=1)

srdata = exp_acc[exp_acc['group'] == 'sr']
srdata = srdata.drop(['group', 'condition'], axis=1)
alldata = exp_acc.drop(['group', 'condition'], axis=1)

# expt 1: 0.25; expt 2: 0.50; expt 3 : 0.3333
bfs = [pg.ttest(alldata.values[:, i], 0.3333,
                alternative="greater")['BF10'][0] for i in range(alldata.shape[1])]
bfs = [np.log10(float(num)) for num in bfs]
int_bfs = [interpret_bayes_factor_log(num) for num in bfs]

# now add evidence for group differences
bfs_gc = [pg.ttest(srdata.values[:, i], controldata.values[:, i],
                   alternative="two-sided")['BF10'][0] for i in range(alldata.shape[1])]
bfs_gc = [np.log10(float(num)) for num in bfs_gc]
int_bfs_gc = [interpret_bayes_factor_log(num) for num in bfs_gc]

# If bfs_gc[i] is positive there is evidence in favor of the alternative hypothesis
# for the ii - th comparison (i.e., the SR group is different from the control group in that measurement).

# If bfs_gc[i] is negative there is evidence in favor of the null hypothesis
# for the ii - th comparison (i.e., there is no significant difference between the SR group and the control group in that measurement).

# If bfs_gc[i] is close to 0 the data do not provide strong evidence
# for either hypothesis for the ii-th comparison.

freq_vector = np.arange(1, 48)
alldata = alldata.drop(columns=['48'])
controldata = controldata.drop(columns=['48'])
srdata = srdata.drop(columns=['48'])

plt.figure(figsize=(18 * cms, 6 * cms))
# plt.suptitle('Experiment 3')
plt.scatter(freq_vector, alldata.values.mean(0), s=5, facecolors='k', edgecolors='k')
plt.errorbar(freq_vector, alldata.values.mean(0), yerr=alldata.values.std(0) / np.sqrt(len(alldata)),
             capsize=0.7, color='k', elinewidth=0.9, linewidth=0.5)

plt.plot(freq_vector, srdata.values.mean(0), color='red', alpha=0.7, linestyle='--')
plt.plot(freq_vector, controldata.values.mean(0), color='blue', alpha=0.7, linestyle='--')

# plt.scatter(6*[freq_vector + 0.2], srdata.values, color='red', s=0.5)
# plt.scatter(8*[freq_vector - 0.2], controldata.values, color='blue', s=0.5)
plt.plot([-1, 100], [0.33, 0.33], linestyle='--', color='green', alpha=0.9)
# Plot each point with the corresponding color and add patches
for i in range(47):
    # Add the patch with the corresponding color
    color = category_colors[int_bfs_gc[i]]
    rect = mpatches.Rectangle((freq_vector[i] - 0.3, 0.34), 0.6, 1, color=color, alpha=1, fill='none')
    plt.gca().add_patch(rect)
    color = category_colors[int_bfs[i]]
    rect = mpatches.Rectangle((freq_vector[i] - 0.3, 0), 0.6, 0.32, color=color, alpha=1, fill='none')
    plt.gca().add_patch(rect)

plt.xlabel('frequency')
plt.ylabel("accuracy")
plt.xlim(0.5, 20.5)
plt.xlim(0.5, 47.5)
plt.ylim(0.28, 0.79)  # Adjust according to your data range
plt.legend(['average', 'super recognizers', 'controls'], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=8)

# Create a custom colormap with transparency
colors_with_alpha = [(r, g, b, 1) for r, g, b, _ in [mcolors.to_rgba(category_colors[cat]) for cat in categories]]  # Extract RGB values and add alpha
cmap_custom = mcolors.ListedColormap(colors_with_alpha)

# Create a colorbar with adjusted position
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
bounds = np.arange(len(categories) + 1)
norm = mcolors.BoundaryNorm(bounds, len(categories))
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_custom), cax=cax, ticks=np.arange(len(categories)) + 0.5)
cb.set_ticklabels(['< -2', '-2 to -1.5', '-1.5 to -1', '-1 to -0.5', '-0.5 to 0.5', ' 0.5 to 1', ' 1 to 1.5', ' 1.5 to 2', ' > 2'])
# Add title to the colorbar
cb.ax.set_title(r'Log($BF_{10}$)', fontsize=10, pad=10, loc='left')

plt.show()
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3.jpg'), dpi=500)

# Plot individual subject data upper and lower panel of the figure
# 3 controls and 3 super recognizers to illustrate variability
# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
plt.figure(figsize=(18 * cms, 3 * cms))
# results from experiment
results_dir = os.path.join(main_dir, 'results', 'classification_data')
results = list(filter(lambda s: '_acc_expt_3.npy' in s, os.listdir(results_dir)))
idx_sr = [5, 9, 13]
idx_con = [4, 3, 11]
ixp = 0
delticks = 1
# change idx_sr to idx_con. SRs are 3_2
# change color 'red' to 'blue' for controls
for id_ in idx_sr:

    subj = results[id_]
    subj_id = subj.replace('_acc_expt_3.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    group_ = demog_file.loc[subj_id].Label
    plt.subplot(1, 3, ixp + 1)
    ixp = ixp + 1
    # true data
    prc_ = np.percentile(data_, (25, 50, 75), axis=0)
    prc_ = prc_[:, 1:-1]
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]
    plt.scatter(freq_vector, prc_[1], s=5, c='red', alpha=0.7)
    plt.errorbar(freq_vector, prc_[1], yerr=prc_[[0, 2]], capsize=1, color='red', elinewidth=0.5, linewidth=0.6, alpha=0.5)
    plt.xlim(0.5, 47.5)

    plt.xticks([10, 20, 30, 40], [10, 20, 30, 40])
    plt.yticks([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    if delticks == 1:
        plt.xticks([])
        plt.yticks([])
    plt.ylim(0.28, 0.79)  # Adjust according to your data range
    plt.plot([-1, 100], [0.3333, 0.3333], linestyle='--', color='green', alpha=0.4)
    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3_2_dticks.jpg'), dpi=500)

# Create final figure by stacking the generated subplots
image1 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3_2_dticks.jpg'))
image2 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3.jpg'))
image3 = Image.open(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_3_3_dticks.jpg'))

# Get dimensions of the images
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size

# Calculate the cropping dimensions
crop_height = int(height1 * 0.1)  # 10% of the height of image1

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
combined_image.save(os.path.join(main_dir, 'results', 'figures', 'Experiment_3.jpg'))
combined_image.show()
