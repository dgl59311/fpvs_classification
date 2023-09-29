# plot and generate results
import os
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from funcs.dependencies_clf import experiment_info

matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'classification_data')

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# set to 1 to plot random label permuted data
plot_perm = 1
cm = 1/2.54  # centimeters in inches

# results from experiment
results = list(filter(lambda s: 'experiment_2.npy' in s, os.listdir(results_dir)))
rlp = list(filter(lambda s: 'experiment_2rlp' in s, os.listdir(results_dir)))
sdt = list(filter(lambda s: 'experiment_2tp_fp_fn_tn' in s, os.listdir(results_dir)))

data_cat = pd.DataFrame()
for id_ in range(14):

    subj = results[id_]
    subj_id = subj.replace('_experiment_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    rlp_ = np.load(os.path.join(results_dir, rlp[id_]))
    sdt_ = np.load(os.path.join(results_dir, sdt[id_]))
    group_ = demog_file.loc[subj_id].Label

    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    # create dataframes
    prc_ = np.percentile(data_[:, 0, :], (25, 50, 75), axis=0)
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]

    df_tmp = pd.DataFrame([prc_[1]])
    df_tmp.columns = freq_vector + 1
    df_tmp.index = [subj_id]
    df_tmp['group'] = g_id

    if id_ == 0:
        data_cat = df_tmp
    else:
        data_cat = pd.concat([data_cat, df_tmp])

data_cat.to_csv(os.path.join(main_dir, 'results', 'sensitivity', 'Experiment_2.csv'))

# get d prime confidence intervals
repetitions = 5000
control = data_cat[data_cat['group'] == 'control']
sr = data_cat[data_cat['group'] == 'sr']
con_true = [control[i].mean() for i in range(1, 49)]
sr_true = [sr[i].mean() for i in range(1, 49)]
eff_size_true = [pg.compute_effsize(control[i], sr[i]) for i in range(1, 49)]
pval_true = [pg.ttest(control[i], sr[i])['p-val'][0] for i in range(1, 49)]
shuff_d = np.zeros((repetitions, len(freq_vector)))

for nr in range(repetitions):
    shuffled_column = np.random.permutation(data_cat['group'])
    data_cat.loc[:, 'group'] = shuffled_column
    control = data_cat[data_cat['group'] == 'control']
    sr = data_cat[data_cat['group'] == 'sr']
    diff = [sr[i].mean() - control[i].mean() for i in range(1, 49)]
    shuff_d[nr, :] = np.array(diff)

plt.figure(figsize=(18 * cm, 13 * cm))
plt.suptitle('Experiment 2')
ci_ = np.abs(np.percentile(shuff_d, (2.5, 97.5), axis=0))
plt.scatter(freq_vector + 1 - 0.1, con_true, s=30, facecolors='none', edgecolors='blue')
plt.errorbar(freq_vector + 1 - 0.1, con_true, yerr=ci_, capsize=2, color='blue', elinewidth=0.5, linewidth=0.7)
plt.scatter(freq_vector + 1 + 0.1, sr_true, s=30, facecolors='none', edgecolors='red')
plt.errorbar(freq_vector + 1 + 0.1, sr_true, yerr=ci_, capsize=2, color='red', elinewidth=0.5, linewidth=0.7)

plt.legend(['controls', 'SRs'])
plt.xlabel('frequency')
plt.ylabel("classifier sensitivity d'")

plt.ylim([-0.4, 1.6])
# Highlight the significant point with a transparent square
highlight_size = 0.5
highlight_color = 'Gray'
highlight_alpha = 0.3

significant_ = freq_vector[np.where(np.array(pval_true) < 0.05)[0]]
for i in range(len(significant_)):
    significant_y = -0.15
    significant_x = significant_[i] + 1
    plt.fill_between([significant_x - highlight_size/2, significant_x + highlight_size/2],
                     plt.ylim()[0], plt.ylim()[1],
                     color=highlight_color, alpha=highlight_alpha)
    plt.text(significant_x, significant_y, 'cohen d = ' + str(eff_size_true[significant_[i - 1]].round(1)),
             rotation=90, horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.legend(['controls', 'SRs', 'significant w/o correction'])
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_2' + '.jpg'), dpi=500)


# Plot individual subject data
plt.figure(figsize=(18*cm, 18*cm))
plt.suptitle('Experiment 2 - 1')
for id_ in range(9):

    subj = results[id_]
    subj_id = subj.replace('_experiment_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    rlp_ = np.load(os.path.join(results_dir, rlp[id_]))
    group_ = demog_file.loc[subj_id].Label

    if id_ < 9:
        id_pl = id_
    elif id_ < 18:
        id_pl = id_ - 9
    else:
        id_pl = id_ - 18

    plt.subplot(3, 3, id_pl + 1)
    # true data
    prc_ = np.percentile(data_[:, 0, :], (25, 50, 75), axis=0)
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]
    plt.scatter(freq_vector + 1, prc_[1], s=5, c='Blue')
    plt.errorbar(freq_vector + 1, prc_[1], yerr=prc_[[0, 2]], capsize=2, color='Blue', elinewidth=0.5, linewidth=0.7)
    # perm data
    if plot_perm == 1:
        prc_ = np.percentile(rlp_[:, 0, :], (25, 50, 75), axis=0)
        prc_[0] = prc_[1] - prc_[0]
        prc_[2] = prc_[2] - prc_[1]
        plt.scatter(freq_vector + 1, prc_[1], s=5, c='k')
        plt.errorbar(freq_vector + 1, prc_[1], yerr=prc_[[0, 2]], capsize=2, color='k', elinewidth=0.5, linewidth=0.7)

    plt.ylabel("d'", labelpad=0.01)
    plt.xlabel('frequency')
    plt.ylim([-0.5, 2])
    plt.yticks([0, 1, 2], [0, 1, 2])
    plt.xlim(-1, 50)
    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    plt.text(-0.5, 1.75, g_id)
    plt.legend(['data', 'random labels'], loc="upper right", ncol=1, fontsize=6)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_2_' + 'subj_1-9' + '.jpg'), dpi=500)

