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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.base import clone
from sklearn.multiclass import OneVsOneClassifier
from funcs.dependencies_clf import data_check, find_headers, experiment_info, dprime_clf
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations

matplotlib.use('Qt5Agg')
patch_sklearn()
main_dir = os.getcwd()

# Classifier
# the entire procedure is repeated 'repeat_n' times
repeat_n = 10
# define random split function: 75% of the trials will be used to train the classifier
random_sp = StratifiedShuffleSplit(n_splits=repeat_n, test_size=0.25, random_state=234)

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

# experiments with more than 2 classes
experiments = [1, 3]
experiments = [1]
for expt in experiments:
    print('Runnin code for experiment: ', expt)
    conditions, clf_files_local, _fname_ = experiment_info(expt)
    files_ = data_check(conditions, clf_files_local)
    files_ = files_[files_.index.isin(demog_file.index)]
    participants = files_.index

    participants = participants[3:4]

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
        conf_mat_mc = np.zeros((repeat_n, len(conditions), len(conditions), len(freq_vector)))
        acc_master = np.zeros((repeat_n, len(freq_vector)))
        for train_i, test_i in tqdm(random_sp.split(eeg_data, cat_labels), total=random_sp.get_n_splits()):
            eeg_train = eeg_data[train_i]
            cat_train = cat_labels[train_i]
            eeg_test = eeg_data[test_i]
            cat_test = cat_labels[test_i]
            for f_i in range(len(freq_vector)):
                # fit OVO classifier
                ovo_internal = clone(clf_model_ovo)
                fit_ovo_model = ovo_internal.fit(eeg_train[:, :, f_i], cat_train)
                ovo_prediction = fit_ovo_model.predict(eeg_test[:, :, f_i])
                # quantify multiclass_
                conf_mat_mc[rep_, :, :, f_i] = confusion_matrix(cat_test, ovo_prediction, labels=conditions)
                acc_master[rep_, f_i] = accuracy_score(ovo_prediction, cat_test)
                # go through estimators
                # see class OneVsOneClassifier, fit()
                k = 0
                class_pairs = []
                for ii_ in range(len(conditions)):
                    for jj_ in range(ii_ + 1, len(conditions)):
                        curr_est = fit_ovo_model.estimators_[k]
                        cond = np.logical_or(cat_test == conditions[ii_],
                                             cat_test == conditions[jj_])
                        int_prediction = curr_est.predict(eeg_test[cond, :, f_i])
                        int_prediction = int_prediction.astype(dtype=str)
                        int_prediction[int_prediction == '0'] = conditions[ii_]
                        int_prediction[int_prediction == '1'] = conditions[jj_]
                        true_values = cat_test[cond]
                        tn, fp, fn, tp = confusion_matrix(true_values, int_prediction).ravel()
                        # d_primes[rep_, k, f_i] = dprime_clf(tp, fp, fn, tn)
                        d_primes[rep_, k, f_i] = accuracy_score(true_values, int_prediction)
                        k = k + 1
                        class_pairs.append((conditions[ii_], conditions[jj_]))
            rep_ = rep_ + 1

        # save results from each participant
        np.save(os.path.join(main_dir, 'results', 'classification_data',
                             participants[i] + '_conf_mat_multiclass_expt_' + str(expt) + '.npy'), conf_mat_mc)

        np.save(os.path.join(main_dir, 'results', 'classification_data',
                             participants[i] + '_single_class_dprimes_expt_' + str(expt) + '.npy'), d_primes)

    np.save(os.path.join(main_dir, 'results', 'classification_data',
                         'experiment_' + str(expt) + '_conditions.npy'), class_pairs)
