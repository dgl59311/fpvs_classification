# plot and generate results
import os
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from funcs.dependencies_clf import clf_id, cohen_d, experiment_info, calculate_d_prime

matplotlib.use('Qt5Agg')
pd.options.mode.chained_assignment = None  # default='warn'
main_dir = os.getcwd()
results_dir = os.path.join(main_dir, 'results', 'classification_data')

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# colors
pl_colors = ['r', 'b', 'g']
# set to 1 to plot random label permuted data
plot_perm = 0
cm = 1/2.54  # centimeters in inches
data_cat = pd.DataFrame()

for expt in range(3, 4):
    # results from experiment
    id_files = 'ovo_experiment_3'
    results = list(filter(lambda s: id_files + '.npy' in s, os.listdir(results_dir)))
    conditions, clf_files_local, _fname_ = experiment_info(expt)
    for id_ in range(2):
        subj = results[id_]
        subj_id = subj.replace(id_files + '.npy', '')
        data_ = np.load(os.path.join(results_dir, subj))
        data_d = np.zeros(data_.shape)
        for freq in range(len(freq_vector)):
            for rep in range(100):
                data_d[rep, :, 0, freq] = calculate_d_prime(data_[rep, :, :, freq])
        mean_ = [data_d.mean(0)[:, :, f][np.triu_indices(len(conditions), 1)].mean() for f in range(len(freq_vector))]



