# Classification analysis
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from sklearnex import patch_sklearn
from sklearn.model_selection import StratifiedShuffleSplit, permutation_test_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.multiclass import OneVsOneClassifier
from funcs.dependencies_clf import data_check, find_headers, clf_fpvs, experiment_info
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
patch_sklearn()
main_dir = os.getcwd()

# Classifier
# the entire procedure is repeated 'repeat_n' times
repeat_n = 100
# define random split function: 75% of the trials will be used to train the classifier
random_sp = StratifiedShuffleSplit(n_splits=repeat_n, test_size=0.25, random_state=234)
random_sp_rlp = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=234)

# classification model
# In OnevsOne the class that received most votes is selected
clf_model_mc = Pipeline(steps=[('scale', StandardScaler()),
                               ('pca', PCA(n_components=20)),
                               ('clf_model', OneVsOneClassifier(LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                                                     penalty='l1',
                                                                                     max_iter=200,
                                                                                     cv=5,
                                                                                     tol=1e-3,
                                                                                     solver='liblinear',
                                                                                     n_jobs=-1)))])
clf_model_bin = Pipeline(steps=[('scale', StandardScaler()),
                                ('pca', PCA(n_components=20)),
                                ('clf_model', LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                                   penalty='l1',
                                                                   max_iter=200,
                                                                   cv=5,
                                                                   tol=1e-3,
                                                                   solver='liblinear',
                                                                   n_jobs=-1))])

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)

# Select which experiment to run
# 1: Identity oddball
# 2: Category selectivity
# 3: Duty Cycle
for expt in range(3, 4):

    print('Runnin code for experiment: ', expt)
    conditions, clf_files_local, id_files = experiment_info(expt)
    files_ = data_check(conditions, clf_files_local)
    participants = files_.index
    participants = participants[9:]

    if len(conditions) > 2:
        clf_model = clf_model_mc
        print('Multiclass classification')
    else:
        clf_model = clf_model_bin
        print('Binary classification')

    # start loop
    for i in tqdm(range(len(participants))):
        eeg_data = []
        cat_labels = []
        d_clf = np.zeros((repeat_n, len(conditions), len(freq_vector)))
        sdt_clf = np.zeros((repeat_n, len(conditions), 4, len(freq_vector)))
        d_clf_rlp = np.zeros((repeat_n, len(conditions), len(freq_vector)))
        for j in range(len(conditions)):
            loc_files = files_.loc[participants[i]]
            # load data for each condition
            data_file = h5py.File(os.path.join(clf_files_local, loc_files[conditions[j]]))
            data_file = data_file['data'][:]
            # get info from the .lw6 files
            [epochs, xstart, xsteps, xlen] = find_headers(os.path.join(clf_files_local, loc_files[conditions[j]]))
            cat_labels.append(np.ones(epochs) * j)
            # select only the first 50 frequencies and 64 electrodes
            eeg_data.append(data_file[0:50, :, :, :, 0:64, :])
        eeg_data = np.concatenate(eeg_data, axis=5)
        # transpose dimensions to trials x electrodes x frequencies
        eeg_data = np.transpose(np.squeeze(eeg_data), (2, 1, 0))
        cat_labels = np.concatenate(cat_labels)
        # for "true" data
        rep_ = 0
        for train_i, test_i in random_sp.split(eeg_data, cat_labels):
            results_clf = clf_fpvs(train_i, test_i, clf_model, eeg_data, cat_labels, freq_vector)
            # returns d prime
            # results_clf[0] = cat x sdt measure x freq
            # results_clf[1] = cat x freq
            sdt_clf[rep_, :, :, :] = results_clf[0]
            d_clf[rep_, :, :] = results_clf[1]
            rep_ = rep_ + 1
        # save data
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + '.npy'), d_clf)
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + 'tp_fp_fn_tn.npy'), d_clf)

        # for random_label permutations
        rep_rlp = 0
        for k in range(repeat_n):
            indices = np.random.permutation(len(cat_labels))
            rlp_labels = cat_labels[indices]
            for train_ri, test_ri in random_sp_rlp.split(eeg_data, rlp_labels):
                results_clf_rlp = clf_fpvs(train_ri, test_ri, clf_model, eeg_data, rlp_labels, freq_vector)
                d_clf_rlp[rep_rlp, :, :] = results_clf_rlp[1]
            rep_rlp = rep_rlp + 1

        # save data
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + 'rlp.npy'), d_clf_rlp)

