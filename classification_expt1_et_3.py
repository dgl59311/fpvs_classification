# Classification analysis
# Experiments 1 and 3
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from sklearnex import patch_sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.base import clone
from sklearn.multiclass import OneVsOneClassifier
from funcs.dependencies_clf import data_check, find_headers, experiment_info
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations

matplotlib.use('Qt5Agg')
patch_sklearn()
main_dir = os.getcwd()

# Classifier
# the entire procedure is repeated 'repeat_n' times
repeat_n = 100
# define random split function: 60% of the trials will be used to train the classifier
random_sp = StratifiedShuffleSplit(n_splits=repeat_n, test_size=0.4, random_state=234)

# classification model
clf_model = Pipeline(steps=[('var', VarianceThreshold()),
                            ('scale', StandardScaler()),
                            ('pca', PCA(n_components=20)),
                            ('clf model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3))])

# OVO classification model
clf_model_ovo = OneVsOneClassifier(clf_model, n_jobs=-1)

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
# only analyze frequencies below 48 Hz / had to be 49
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]

# run for experiments with more than 2 classes
experiments = [1, 3]
for expt in experiments:
    print('Runnin code for experiment: ', expt)
    conditions, clf_files_local, _fname_ = experiment_info(expt)
    files_ = data_check(conditions, clf_files_local)
    files_ = files_[files_.index.isin(demog_file.index)]
    participants = files_.index
    # start loop
    for i in tqdm(range(len(participants))):
        eeg_data = []
        cat_labels = []
        for j in range(len(conditions)):
            loc_files = files_.loc[participants[i]]
            # load .mat data for each condition
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
        # allocate results
        conf_mat_mc = np.zeros((repeat_n, len(conditions), len(conditions), len(freq_vector)))
        f1_mclass = np.zeros((repeat_n, len(freq_vector)))
        acc_mclass = np.zeros((repeat_n, len(freq_vector)))
        # repeated train-test split for unbiased estimates of accuracy with small sample sizes
        # sample 60% of the trials, train the model and test on the remaining 40% ... repeat this 100 times
        rep_ = 0
        for train_i, test_i in tqdm(random_sp.split(eeg_data, cat_labels), total=random_sp.get_n_splits()):
            eeg_train = eeg_data[train_i]
            cat_train = cat_labels[train_i]
            eeg_test = eeg_data[test_i]
            cat_test = cat_labels[test_i]
            # train models per frequency
            for f_i in range(len(freq_vector)):
                # fit OVO classifier
                ovo_internal = clone(clf_model_ovo)
                fit_ovo_model = ovo_internal.fit(eeg_train[:, :, f_i], cat_train)
                ovo_prediction = fit_ovo_model.predict(eeg_test[:, :, f_i])
                # quantify multiclass_
                conf_mat_mc[rep_, :, :, f_i] = confusion_matrix(cat_test, ovo_prediction, labels=conditions)
                f1_mclass[rep_, f_i] = f1_score(cat_test, ovo_prediction, average='macro')
                acc_mclass[rep_, f_i] = accuracy_score(cat_test, ovo_prediction)
            rep_ = rep_ + 1
        # save results from each participant
        np.save(os.path.join(main_dir, 'results', 'classification_data',
                             participants[i] + '_conf_mat_expt_' + str(expt) + '.npy'), conf_mat_mc)

        np.save(os.path.join(main_dir, 'results', 'classification_data',
                             participants[i] + '_f1_expt_' + str(expt) + '.npy'), f1_mclass)

        np.save(os.path.join(main_dir, 'results', 'classification_data',
                             participants[i] + '_acc_expt_' + str(expt) + '.npy'), acc_mclass)
