# Classification analysis
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from sklearnex import patch_sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.base import clone
from sklearn.multiclass import OneVsOneClassifier
from funcs.dependencies_clf import data_check, find_headers, ovo_clf_fpvs, experiment_info, clf_id, dprime_clf
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations


matplotlib.use('Qt5Agg')
patch_sklearn()
main_dir = os.getcwd()

# Classifier
# the entire procedure is repeated 'repeat_n' times
repeat_n = 5
# define random split function: 75% of the trials will be used to train the classifier
random_sp = StratifiedShuffleSplit(n_splits=repeat_n, test_size=0.25, random_state=234)
random_sp_rlp = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=234)

# classification model
clf_model = Pipeline(steps=[('scale', StandardScaler()),
                            ('pca', PCA(n_components=20)),
                            ('clf_model', LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                               penalty='l1',
                                                               max_iter=200,
                                                               cv=10,
                                                               tol=1e-3,
                                                               solver='liblinear',
                                                               n_jobs=-1))])

# OVO classification model
clf_model_ovo = OneVsOneClassifier(clf_model)

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]

for expt in range(1, 2):
    print('Runnin code for experiment: ', expt)
    conditions, clf_files_local, _fname_ = experiment_info(expt)
    files_ = data_check(conditions, clf_files_local)
    files_ = files_[files_.index.isin(demog_file.index)]
    participants = files_.index
    participants = participants[0:1]
    pairwise_clf = list(combinations(conditions, 2))
    # start loop
    for i in tqdm(range(len(participants))):
        eeg_data = []
        cat_labels = []
        for j in range(len(conditions)):
            loc_files = files_.loc[participants[i]]
            # load data for each condition
            data_file = h5py.File(os.path.join(clf_files_local, loc_files[conditions[j]]))
            data_file = data_file['data'][:]
            # get info from the .lw6 files
            [epochs, xstart, xsteps, xlen] = find_headers(os.path.join(clf_files_local, loc_files[conditions[j]]))
            cat_labels.append([conditions[j]] * epochs)
            # select only the first 50 frequencies and 64 electrodes
            eeg_data.append(data_file[0:50, :, :, :, 0:64, :])
        eeg_data = np.concatenate(eeg_data, axis=5)
        # transpose dimensions to trials x electrodes x frequencies
        eeg_data = np.transpose(np.squeeze(eeg_data), (2, 1, 0))
        cat_labels = np.concatenate(cat_labels)
        # for "true" data
        rep_ = 0
        d_primes = np.zeros((repeat_n, len(pairwise_clf), len(freq_vector)))
        all_pred = np.zeros((repeat_n, len(pairwise_clf), 109, len(freq_vector))).astype(dtype=str)
        decision_f = np.zeros((repeat_n, len(pairwise_clf), 109, len(freq_vector)))
        true_labels = np.zeros((repeat_n, 109)).astype(dtype=str)
        for train_i, test_i in random_sp.split(eeg_data, cat_labels):
            eeg_train = eeg_data[train_i]
            cat_train = cat_labels[train_i]
            eeg_test = eeg_data[test_i]
            cat_test = cat_labels[test_i]
            true_labels[rep_, :] = cat_test
            for clf_i in range(len(pairwise_clf)):
                curr_ = pairwise_clf[clf_i]
                #print(curr_)
                i_clf_train = [(cat_train == curr_[0]) | (cat_train == curr_[1])][0]
                i_clf_test = [(cat_test == curr_[0]) | (cat_test == curr_[1])][0]
                tmp_eeg_train = eeg_train[i_clf_train]
                tmp_cat_train = cat_train[i_clf_train]
                tmp_eeg_test = eeg_test[i_clf_test]
                tmp_cat_test = cat_test[i_clf_test]
                for f_i in range(len(freq_vector)):
                    freq_ = freq_vector[f_i]
                    freq_train = tmp_eeg_train[:, :, freq_]
                    freq_test = tmp_eeg_test[:, :, freq_]
                    # fit classifier on train data with cross-validation
                    internal_clf = clone(clf_model)
                    fit_model = internal_clf.fit(freq_train, tmp_cat_train)
                    prediction_ = fit_model.predict(freq_test)
                    tn, fp, fn, tp = confusion_matrix(tmp_cat_test, prediction_).ravel()
                    d_primes[rep_, clf_i, f_i] = dprime_clf(tp, fp, fn, tn)
                    # test for all classes and store predicitions
                    decision_f[rep_, clf_i, :, f_i] = fit_model.decision_function(eeg_test[:, :, f_i])
                    all_pred[rep_, clf_i, :, f_i] = fit_model.predict(eeg_test[:, :, f_i])
                    # fit OVO classifier
                    ovo_internal = clone(clf_model_ovo)
                    fit_ovo_model = ovo_internal.fit(eeg_train[:, :, f_i], cat_train)
                    ovo_prediction = fit_ovo_model.predict(eeg_test[:, :, f_i])

            rep_ = rep_ + 1
            print(rep_)

n_classes = 4
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        print(conditions[i])