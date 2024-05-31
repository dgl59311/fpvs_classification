# plot and generate results
import os
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from funcs.dependencies_clf import cohen_d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from scipy import stats


matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'classification_data')

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]
# results from experiment
acc_results = list(filter(lambda s: 'acc_expt_2.npy' in s, os.listdir(results_dir)))
f1_results = list(filter(lambda s: 'f1_expt_2.npy' in s, os.listdir(results_dir)))
sdt = list(filter(lambda s: 'tp_fp_fn_tn' in s, os.listdir(results_dir)))

# write csv file with median d' for all subjects
data_cat = pd.DataFrame()
for id_ in range(14):
    subj = f1_results[id_]
    subj_id = subj.replace(' f1_expt_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    sdt_ = np.load(os.path.join(results_dir, sdt[id_]))
    group_ = demog_file.loc[subj_id].Label

    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    # create dataframes
    prc_ = np.percentile(data_, (25, 50, 75), axis=0)

    df_tmp = pd.DataFrame([prc_[1]])
    df_tmp.columns = freq_vector + 1
    df_tmp.index = [subj_id]
    df_tmp['group'] = g_id
    df_tmp['condition'] = 'expt2'

    if id_ == 0 == 0:
        data_cat = df_tmp
    else:
        data_cat = pd.concat([data_cat, df_tmp])

data_cat.to_csv(os.path.join(main_dir, 'results', 'sensitivity', 'Experiment_2.csv'))

# get d prime confidence intervals
best_freq = []
best_fx = []
data_conf = pd.DataFrame()

# results from experiment
sdt = list(filter(lambda s: 'tp_fp_fn_tn' in s, os.listdir(results_dir)))
filt_condition = data_cat[data_cat['condition'] == 'expt2']
control = filt_condition[filt_condition['group'] == 'control']
sr = filt_condition[filt_condition['group'] == 'sr']

# take mean d' of each frequency for controls and SR
con_true = [control[i].mean() for i in range(1, 49)]
sr_true = [sr[i].mean() for i in range(1, 49)]

# calculate MOE margin of Error for d' for each group
con_moe = [1.645 * (control[i].std()/np.sqrt(8)) for i in range(1, 49)]
sr_moe = [1.645 * (sr[i].std()/np.sqrt(6)) for i in range(1, 49)]

# lower moe
con_lm = np.array([con_true[i] - con_moe[i] for i in range(len(con_moe))])
con_zdprime = np.where(con_lm < 0)[0]
con_mask = np.ones(len(freq_vector), dtype=bool)
con_mask[con_zdprime] = False

sr_lm = np.array([sr_true[i] - sr_moe[i] for i in range(len(sr_moe))])
sr_zdprime = np.where(sr_lm < 0)[0]
sr_mask = np.ones(len(freq_vector), dtype=bool)
sr_mask[sr_zdprime] = False


# calculate one sample t test vs zero to see which d primes are significant.
# calculate one sample t test vs zero to see which d primes are significant.
# con
#con_true_cohend = [pg.ttest(control[i], 0, alternative='greater')['cohen-d'][0] for i in range(1, 49)]
#con_true_pval = [pg.ttest(control[i], 0, alternative='greater')['p-val'][0] for i in range(1, 49)]
#con_mask = pg.multicomp(con_true_pval, method='bonf')[0]
#con_zdprime = np.where(con_mask == False)[0]
# sr
#sr_true_cohend = [pg.ttest(sr[i], 0, alternative='greater')['cohen-d'][0] for i in range(1, 49)]
#sr_true_pval = [pg.ttest(sr[i], 0, alternative='greater')['p-val'][0] for i in range(1, 49)]
#sr_mask = pg.multicomp(sr_true_pval, method='bonf')[0]
#sr_zdprime = np.where(sr_mask == False)[0]


# compare each frequency between patients and controls
eff_size_true = [cohen_d(sr[i], control[i]) for i in range(1, 49)]
pval_true = [pg.ttest(control[i], sr[i])['p-val'][0] for i in range(1, 49)]
repetitions = 1000
shuff_d = np.zeros((repetitions, len(freq_vector)))

for nr in range(repetitions):
    shuffled_column = np.random.permutation(filt_condition['group'])
    filt_condition.loc[:, 'group'] = shuffled_column
    control = filt_condition[filt_condition['group'] == 'control']
    sr = filt_condition[filt_condition['group'] == 'sr']
    diff = [control[ii].mean() - sr[ii].mean() for ii in range(1, 49)]
    shuff_d[nr, :] = np.array(diff)

ci_ = np.abs(np.percentile(shuff_d, (2.5, 97.5), axis=0))

# concatenate results
con_dprimes = np.array(con_true)
sr_dprimes = np.array(sr_true)
ci_dprimes = np.array(ci_)
eff_sizes = np.array(eff_size_true)
p_values = np.array(pval_true)
freq_vector = np.array(freq_vector)
# obtain average confusion matrix for controls and SR for highest effect size
# look only for data below 20Hz
sorted_fx = np.sort(eff_size_true[0:20])[::-1]
index_1 = np.where(sorted_fx[0] == eff_size_true)[0][0]
print('highest effect size found at:', index_1 + 1, 'Hz')
print('effect size:', np.round(sorted_fx[0], 2))
best_freq.append(index_1 + 1)
best_fx.append(np.round(sorted_fx[0], 2))

largest_sr = 0
if largest_sr == 1:
    # if instead of frequency with the largest effect between groups we want the one where SR had higher accuracy
    best_freq = [np.argmax(sr_dprimes[0]) + 1]
    best_fx = [np.round(eff_size_true[best_freq[0] - 1], 2)]
    index_1 = best_freq[0] - 1


for j in range(len(sdt)):
    subj = sdt[j]
    subj_id = subj.replace('expt_2_tp_fp_fn_tn.npy', '')
    group_ = demog_file.loc[subj_id].Label
    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'
    sdt_ = np.load(os.path.join(results_dir, sdt[j]))
    sdt_ = sdt_[:, 0, :, :]
    sdt_1 = sdt_[:, :, index_1]
    avg_cmat = np.zeros((100, 2, 2))
    for k in range(100):
        tmp_cmat = sdt_1[k].reshape(2, 2)
        tmp_cmat[0, :] = tmp_cmat[0, :] / np.sum(tmp_cmat[0, :])
        tmp_cmat[1, :] = tmp_cmat[1, :] / np.sum(tmp_cmat[1, :])
        avg_cmat[k, :, :] = tmp_cmat
    avg_cmat = np.nanmean(avg_cmat, 0).reshape(1, -1)[0]
    # create dataframe with results
    df_tmp = pd.DataFrame([avg_cmat])
    df_tmp.index = [subj_id]
    df_tmp['group'] = g_id
    df_tmp['condition'] = 'expt2'

    if j == 0:
        data_conf = df_tmp
    else:
        data_conf = pd.concat([data_conf, df_tmp])

f_titles = ['Experiment 2']
# Highlight the significant point with a transparent square
highlight_size = 0.5
highlight_color = 'Gray'
highlight_alpha = 0.3
x_values = np.arange(1, 49)
bbox_props = dict(boxstyle="square,pad=0.3", fc="white", alpha=0.7, ec="black", lw=2)

# Create plots
# colors
pl_colors = ['r', 'b', 'g']
# set to 1 to plot random label permuted data
plot_perm = 0
cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(18 * cm, 9 * cm))
fig.suptitle(f_titles[0], fontsize=10)
# Create a 1x2 grid for the main plot
outer_grid = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
# Main subplot in the first grid
main_subplot = plt.subplot(outer_grid[0])
main_subplot.scatter(freq_vector[con_mask] - 0.1 + 1, con_dprimes[con_mask], s=20, facecolors='none', edgecolors='blue')
main_subplot.scatter(freq_vector[sr_mask] + 0.1 + 1, sr_dprimes[sr_mask], s=20, facecolors='none', edgecolors='red')


main_subplot.scatter(con_zdprime - 0.1 + 1, con_dprimes[con_zdprime], s=10, facecolors='blue', color='blue')
main_subplot.errorbar(freq_vector[con_mask] - 0.1 + 1, con_dprimes[con_mask], yerr=ci_dprimes[0][con_mask],
                      capsize=1, color='blue', elinewidth=0.5, linewidth=0)
main_subplot.plot(freq_vector + 1, con_dprimes, color='Blue', linewidth=0.6)

main_subplot.scatter(sr_zdprime + 0.1 + 1, sr_dprimes[sr_zdprime], s=10, facecolors='red', color='red')
main_subplot.errorbar(freq_vector[sr_mask] + 0.1 + 1, sr_dprimes[sr_mask], yerr=ci_dprimes[0][sr_mask],
                      capsize=1, color='red', elinewidth=0.5, linewidth=0)
main_subplot.plot(freq_vector + 1, sr_dprimes, color='Red', linewidth=0.6)

main_subplot.fill_between([best_freq[0] - highlight_size/2, best_freq[0] + highlight_size/2], -0.4, 1.9, color=highlight_color, alpha=highlight_alpha)
plt.text(best_freq[0], 1.6, "Cohen's d = " + str(best_fx[0]), horizontalalignment='left', verticalalignment='center', bbox=bbox_props)
main_subplot.legend(['controls', 'SRs'])
main_subplot.set_xlabel('frequency')
main_subplot.set_xticks(x_values[2::3])
main_subplot.set_ylabel("classifier sensitivity d'")
main_subplot.set_ylim([-0.4, 1.9])
main_subplot.set_title("Classification d prime", fontsize=10)

# Create a 2x1 grid for subplots within the second subplot
inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], hspace=0.5)
# Subplot 1 in the second subplot
sub1 = plt.Subplot(fig, inner_grid[0])
sub1.set_title("SRs (" + str(best_freq[0]) + 'Hz)', fontsize=10)
fig.add_subplot(sub1)
sr = data_conf[data_conf['group'] == 'sr']
sr = sr[sr['condition'] == 'expt2']
data = sr.iloc[:, 0:4].values.mean(0).reshape(2, 2) * 100
img = sub1.imshow(data, cmap='Blues', vmin=0.0, vmax=100)
# Add a colorbar
cbar = plt.colorbar(img)
cbar.set_label('%')
sub1.set_yticks([0, 1])
sub1.set_yticklabels(['faces', 'houses'])
sub1.set_xticks([])
sub1.set_xticklabels([])
sub1.tick_params(axis='y', length=0)
for ii in range(2):
    for jj in range(2):
        sub1.text(jj, ii, f'{data[ii, jj]:.2f}', ha='center', va='center', color='black', fontsize=8)

# Subplot 2 in the second subplot
sub2 = plt.Subplot(fig, inner_grid[1])
sub2.set_title("Controls (" + str(best_freq[0]) + 'Hz)', fontsize=10)
fig.add_subplot(sub2)
con = data_conf[data_conf['group'] == 'control']
con = con[con['condition'] == 'expt2']
data = con.iloc[:, 0:4].values.mean(0).reshape(2, 2) * 100
img = sub2.imshow(data, cmap='Blues', vmin=0.0, vmax=100)
# Add a colorbar
cbar = plt.colorbar(img)
cbar.set_label('%')
sub2.set_yticks([0, 1])
sub2.set_yticklabels(['faces', 'houses'])
sub2.set_xticks([])
sub2.set_xticklabels([])
sub2.tick_params(axis='y', length=0)
for ii in range(2):
    for jj in range(2):
        sub2.text(jj, ii, f'{data[ii, jj]:.2f}', ha='center', va='center', color='black', fontsize=8)
# Finally, show the figure
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(main_dir, 'results', 'figures', 'Experiment_2_group' + '.jpg'), dpi=500)


# Plot individual subject data
plt.figure(figsize=(18*cm, 18*cm))
plt.suptitle('Experiment 2 - 2')
conditions = ['faces', 'houses']
# results from experiment
results = list(filter(lambda s: '_f1_expt_2.npy' in s, os.listdir(results_dir)))
sdt = list(filter(lambda s: '_tp_fp_fn_tn' in s, os.listdir(results_dir)))

for id_ in range(9):

    subj = results[id_]
    subj_id = subj.replace('_f1_expt_2.npy', '')
    data_ = np.load(os.path.join(results_dir, subj))
    subj_confmat = np.load(os.path.join(results_dir, sdt[id_]))
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
    prc_[0] = prc_[1] - prc_[0]
    prc_[2] = prc_[2] - prc_[1]
    plt.scatter(freq_vector + 1, prc_[1], s=5, c='Blue')
    plt.errorbar(freq_vector + 1, prc_[1], yerr=prc_[[0, 2]], capsize=2, color='Blue', elinewidth=0.5, linewidth=0.7)
    ix = np.argmax(prc_[1])
    to_plot = subj_confmat[:, :, ix].mean(0).reshape(2, 2)
    ix = ix + 1
    plt.fill_between([ix - highlight_size / 2, ix + highlight_size / 2], -0.5, 4,
                     color=highlight_color, alpha=highlight_alpha)
    for kk in range(len(to_plot)):
        to_plot[kk, :] = to_plot[kk, :] / np.sum(to_plot[kk, :])
    to_plot = to_plot * 100

    plt.ylabel("F1-score")
    plt.xlabel('frequency')
    plt.ylim([0.4, 0.75])
    plt.yticks([0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8])
    plt.xlim(-1, 50)
    if 'Control' in group_:
        g_id = 'control'
    else:
        g_id = 'sr'

    plt.text(-0.5, 0.75, g_id)

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
