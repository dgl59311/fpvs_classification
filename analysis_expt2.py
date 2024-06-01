# generate results
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


matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'classification_data')

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 49]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]

# results from experiment
acc_results = list(filter(lambda s: '_acc_expt_2.npy' in s, os.listdir(results_dir)))
f1_results = list(filter(lambda s: '_f1_expt_2.npy' in s, os.listdir(results_dir)))

# write csv file with median accuracy across 100 train-test splits for each subject
data_cat = pd.DataFrame()
# consider only frequencies starting from 1 Hz
freq_vector_pl = freq_vector[1:]

for id_ in range(demog_file.shape[0]):
    # this can be run also with the f1 score results
    subj = acc_results[id_]
    subj_id = subj.replace('_acc_expt_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    group_ = demog_file.loc[subj_id].Label

    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    # create dataframes
    prc_ = np.percentile(data_, (25, 50, 75), axis=0)
    prc_ = prc_[:, 1:]
    # take only the median
    df_tmp = pd.DataFrame([prc_[1]])
    df_tmp.columns = freq_vector_pl
    df_tmp.index = [subj_id]
    df_tmp['group'] = g_id
    df_tmp['condition'] = 'expt2'

    if id_ == 0:
        data_cat = df_tmp
    else:
        data_cat = pd.concat([data_cat, df_tmp])

# write csv files with accuracy values for each subject
data_cat.to_csv(os.path.join(main_dir, 'results', 'sensitivity', 'Experiment_2_accuracy.csv'))

# Plot individual subject data
f_titles = ['Experiment 2']
# Highlight the significant point with a transparent square
highlight_size = 0.5
highlight_color = 'Gray'
highlight_alpha = 0.3
x_values = np.arange(1, 48)
bbox_props = dict(boxstyle="square,pad=0.3", fc="white", alpha=0.7, ec="black", lw=2)

# Create plots
# colors
pl_colors = ['r', 'b', 'g']
# set to 1 to plot random label permuted data
plot_perm = 0
cm = 1/2.54  # centimeters in inches

# Plot individual subject data
plt.figure(figsize=(18*cm, 18*cm))
plt.suptitle('Experiment 2 - 2')
# double check
conditions = ['houses', 'faces']
# results from experiment
results = list(filter(lambda s: '_acc_expt_2.npy' in s, os.listdir(results_dir)))
sdt = list(filter(lambda s: '_conf_mat_expt_2.npy' in s, os.listdir(results_dir)))

# consider frequencies until 47 Hz
freq_vector_pl = freq_vector[1:-1]
# either range(9) for Experiment 2 - 1; or range(9, 14) for Experiment 2 - 2
# There can only be 9 subjects per figure
for id_ in range(9, 14):

    subj = results[id_]
    subj_id = subj.replace('_acc_expt_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))[:, :-1]
    subj_confmat = np.load(os.path.join(results_dir, sdt[id_]))[:, :, 1:-1]
    group_ = demog_file.loc[subj_id].Label

    if id_ < 9:
        id_pl = id_
    elif id_ < 18:
        id_pl = id_ - 9
    else:
        id_pl = id_ - 18

    plt.subplot(3, 3, id_pl + 1)

    # true data
    prc_ = np.percentile(data_, (25, 50, 75), axis=0)
    prc_ = prc_[:, 1:]
    # get confidence intervals, 25th and 75th percentiles
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]
    # plot median accuracy
    plt.scatter(freq_vector_pl, prc_[1], s=5, c='Blue')
    # plot error bars
    plt.errorbar(freq_vector_pl, prc_[1], yerr=prc_[[0, 2]], capsize=2, color='Blue', elinewidth=0.5, linewidth=0.7)
    # highlight the frequency with the highest accuracy
    ix = np.argmax(prc_[1])
    # plot the confusion matrix for that frequency
    to_plot = subj_confmat[:, :, ix].mean(0).reshape(2, 2)
    ix = ix + 1
    plt.fill_between([ix - highlight_size / 2, ix + highlight_size / 2], -0.5, 4,
                     color=highlight_color, alpha=highlight_alpha)
    # plot normalized confusion matrix for the frequency with the highest accuracy
    for kk in range(len(to_plot)):
        to_plot[kk, :] = to_plot[kk, :] / np.sum(to_plot[kk, :])
    to_plot = to_plot * 100

    plt.ylabel("accuracy")
    plt.xlabel('frequency')
    plt.ylim([0.43, 0.9])
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlim(-2, 52)
    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    plt.text(-0.5, 0.85, g_id)

    # I want this imshow to appear in the upper right corner inside the plot
    axins = inset_axes(plt.gca(), width="25%", height="25%", loc='upper right')
    img = axins.imshow(to_plot, cmap='Blues', vmin=0, vmax=100)  # Adjust limits as needed

    # Add horizontal colorbar below imshow
    cax = inset_axes(plt.gca(), width="100%", height="10%", loc='lower center', borderpad=-0.5)
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', ticks=[10, 50, 90])
    cbar.set_label('%', fontsize=6, labelpad=1)  # Set label for the colorbar, change as needed
    # Adjust the fontsize of the colorbar tick labels
    cbar.ax.tick_params(axis='x', labelsize=6)
    # Customize the inset_axes if necessary
    axins.set_xticks([])
    axins.set_yticks([0, 1])
    axins.set_yticklabels(conditions, fontsize=7)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_2_' + 'subj_10-14' + '.jpg'), dpi=500)

