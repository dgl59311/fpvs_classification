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
cmap = plt.get_cmap('RdBu_r')
categories = ["Decisive (null)", "Strong 2 (null)", "Strong (null)", "Substantial (null)", "Barely worth mentioning",
              "Substantial", "Strong", "Strong 2", "Decisive"]
category_colors = {category: cmap(i / (len(categories) - 1)) for i, category in enumerate(categories)}

# load data
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'sensitivity')

# Experiment
exp_acc = pd.read_csv(os.path.join(results_dir, 'Experiment_2_accuracy.csv'), index_col=0)
controldata = exp_acc[exp_acc['group'] == 'control']
controldata = controldata.drop(['group', 'condition'], axis=1)

srdata = exp_acc[exp_acc['group'] == 'sr']
srdata = srdata.drop(['group', 'condition'], axis=1)
alldata = exp_acc.drop(['group', 'condition'], axis=1)

# expt 1: 0.25; expt 2: 0.50; expt 3 : 0.3333
bfs = [pg.ttest(alldata.values[:, i], 0.5,
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

# Plot mid pannel of the figure

plt.figure(figsize=(18 * cms, 6 * cms))
# plt.suptitle('Experiment 3')
plt.plot(freq_vector - 0.2, controldata.values.mean(0), color='grey', alpha=1, linestyle='-')
plt.plot(freq_vector + 0.2, srdata.values.mean(0), color='k', alpha=1, linestyle='-')

plt.scatter(freq_vector - 0.2, controldata.values.mean(0), s=5, facecolors='grey', edgecolors='grey', zorder=3)
plt.errorbar(freq_vector - 0.2, controldata.values.mean(0), yerr=controldata.values.std(0) / np.sqrt(len(controldata)),
             capsize=0.7, color='grey', elinewidth=0.9, linewidth=0.5)
# sr data
plt.scatter(freq_vector + 0.2, srdata.values.mean(0), s=5, facecolors='k', edgecolors='k', zorder=2)
plt.errorbar(freq_vector + 0.2, srdata.values.mean(0), yerr=srdata.values.std(0) / np.sqrt(len(srdata)),
             capsize=0.7, color='k', elinewidth=0.9, linewidth=0.5)


# plt.scatter(6*[freq_vector + 0.2], srdata.values, color='red', s=0.5)
# plt.scatter(8*[freq_vector - 0.2], controldata.values, color='blue', s=0.5)
plt.plot([-1, 100], [0.5, 0.5], linestyle='--', color='green', alpha=0.9)
# Plot each point with the corresponding color and add patches
for i in range(len(freq_vector)):
    # Add the patch with the corresponding color
    color = category_colors[int_bfs_gc[i]]
    rect = mpatches.Rectangle((freq_vector[i] - 0.3, 0.51), 0.6, 1, color=color, alpha=1, fill='none')
    plt.gca().add_patch(rect)
    color = category_colors[int_bfs[i]]
    rect = mpatches.Rectangle((freq_vector[i] - 0.3, 0), 0.6, 0.49, color=color, alpha=1, fill='none')
    plt.gca().add_patch(rect)

plt.xlabel('frequency')
plt.ylabel("accuracy")
plt.xlim(0.5, 20.5)
plt.xlim(0.5, 47.5)
plt.ylim(0.44, 0.78)  # Adjust according to your data range
# plt.legend(['average', 'super recognizers', 'controls'], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=8)
plt.legend(['controls', 'super recognizers'], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=8)
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
# cb.set_ticklabels(categories)
cb.set_ticklabels(['< -2', '-2 to -1.5', '-1.5 to -1', '-1 to -0.5', '-0.5 to 0.5', ' 0.5 to 1', ' 1 to 1.5', ' 1.5 to 2', ' > 2'])
# Add title to the colorbar
cb.ax.set_title(r'Log($BF_{10}$)', fontsize=10, pad=10, loc='left')

plt.show()
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Group_Experiment_2_v2.pdf'), dpi=500)
