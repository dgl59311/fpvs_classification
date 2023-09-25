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
conditions, clf_files_local, id_files = experiment_info(3)

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# colors
pl_colors = ['r', 'b', 'g', 'm']
# set to 1 to plot random label permuted data
plot_perm = 0
cm = 1/2.54  # centimeters in inches

# results from experiment
results = list(filter(lambda s: 'experiment_3.npy' in s, os.listdir(results_dir)))
rlp = list(filter(lambda s: 'experiment_3rlp' in s, os.listdir(results_dir)))
sdt = list(filter(lambda s: 'experiment_3tp_fp_fn_tn' in s, os.listdir(results_dir)))

for id_ in range(26):

    subj = results[id_]
    subj_id = subj.replace('_experiment_3.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    rlp_ = np.load(os.path.join(results_dir, rlp[id_]))
    sdt_ = np.load(os.path.join(results_dir, sdt[id_]))
    group_ = demog_file.loc[subj_id].Label

    for j in range(len(conditions)):
        if 'Control' in group_:
            g_id = 'control'
        else:
            g_id = 'sr'

        # create dataframes
        prc_ = np.percentile(data_[:, j, :], (25, 50, 75), axis=0)
        prc_[0] = prc_[1] - prc_[0]
        prc_[2] = prc_[2] - prc_[1]

        df_tmp = pd.DataFrame([prc_[1]])
        df_tmp.columns = freq_vector + 1
        df_tmp.index = [subj_id]
        df_tmp['group'] = g_id
        df_tmp['condition'] = conditions[j]

        if id_ == 0 and j == 0:
            data_cat = df_tmp
        else:
            data_cat = pd.concat([data_cat, df_tmp])

data_cat.to_csv(os.path.join(main_dir, 'results', 'sensitivity', 'Experiment_3.csv'))

# get d prime confidence intervals
for i in range(len(conditions)):
    plt.figure(figsize=(15 * cm, 9 * cm))
    plt.suptitle(conditions[i])
    repetitions = 1000
    filt_condition = data_cat[data_cat['condition'] == conditions[i]]
    control = filt_condition[filt_condition['group'] == 'control']
    sr = filt_condition[filt_condition['group'] == 'sr']
    con_true = [control[i].mean() for i in range(1, 49)]
    sr_true = [sr[i].mean() for i in range(1, 49)]
    eff_size_true = [pg.compute_effsize(control[i], sr[i]) for i in range(1, 49)]
    pval_true = [pg.ttest(control[i], sr[i])['p-val'][0] for i in range(1, 49)]
    shuff_d = np.zeros((repetitions, len(freq_vector)))

    for nr in range(repetitions):
        shuffled_column = np.random.permutation(filt_condition['group'])
        filt_condition.loc[:, 'group'] = shuffled_column
        control = filt_condition[filt_condition['group'] == 'control']
        sr = filt_condition[filt_condition['group'] == 'sr']
        diff = [control[i].mean() - sr[i].mean() for i in range(1, 49)]
        shuff_d[nr, :] = np.array(diff)

    ci_ = np.abs(np.percentile(shuff_d, (2.5, 97.5), axis=0))
    plt.scatter(freq_vector + 1 - 0.1, con_true, s=30, facecolors='none', edgecolors='blue')
    plt.errorbar(freq_vector + 1 - 0.1, con_true, yerr=ci_, capsize=2, color='blue', elinewidth=0.5, linewidth=0.7)
    plt.scatter(freq_vector + 1 + 0.1, sr_true, s=30, facecolors='none', edgecolors='red')
    plt.errorbar(freq_vector + 1 + 0.1, sr_true, yerr=ci_, capsize=2, color='red', elinewidth=0.5, linewidth=0.7)

    plt.legend(['controls', 'SRs'])
    plt.xlabel('frequency')
    plt.ylabel("classifier sensitivity d'")

    plt.ylim([-0.4, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_3_' + conditions[i] + '.jpg'), dpi=500)

# Plot individual subject data
plt.figure(figsize=(18*cm, 18*cm))
plt.suptitle('Experiment 3 - 1')
for id_ in range(9):

    subj = results[id_]
    subj_id = subj.replace('_experiment_3.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    rlp_ = np.load(os.path.join(results_dir, rlp[id_]))
    group_ = demog_file.loc[subj_id].Label

    for j in range(len(conditions)):
        if id_ < 9:
            id_pl = id_
        else:
            id_pl = id_ - 9
        plt.subplot(3, 3, id_pl + 1)
        # true data
        prc_ = np.percentile(data_[:, j, :], (25, 50, 75), axis=0)
        prc_[0] = prc_[1] - prc_[0]
        prc_[2] = prc_[2] - prc_[1]
        plt.scatter(freq_vector + 1, prc_[1], s=5, c=pl_colors[j])
        plt.errorbar(freq_vector + 1, prc_[1], yerr=prc_[[0, 2]], capsize=2, color=pl_colors[j], elinewidth=0.5, linewidth=0.7)
        # perm data
        if plot_perm == 1:
            prc_ = np.percentile(rlp_[:, j, :], (25, 50, 75), axis=0)
            prc_[0] = prc_[1] - prc_[0]
            prc_[2] = prc_[2] - prc_[1]
            plt.scatter(freq_vector + 1, prc_[1], s=5, c=pl_colors[j])
            plt.errorbar(freq_vector + 1, prc_[1], yerr=prc_[[0, 2]], capsize=2, color=pl_colors[j], elinewidth=0.5, linewidth=0.7)

        plt.ylabel("d'")
        plt.xlabel('frequency')
        plt.ylim([-0.5, 3])
        plt.xlim(-1, 50)
        if 'Control' in group_:
            g_id = 'control'
        else:
            g_id = 'sr'

        plt.text(-0.5, 2.5, g_id)

    plt.legend(conditions, loc="upper right", ncol=1, fontsize=7)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_3_' + 'subj_1-9' + '.jpg'), dpi=500)
