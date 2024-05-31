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
from sklearn.multiclass import OneVsOneClassifier
from funcs.dependencies_clf import data_check, find_headers, ovo_clf_fpvs, experiment_info, clf_id
from funcs.custom_classifier import OneVsOneClassifierCustom
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
clf_model = OneVsOneClassifierCustom(Pipeline(steps=[('scale', StandardScaler()),
                                                     ('pca', PCA(n_components=20)),
                                                     ('clf_model', LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                                                        penalty='l1',
                                                                                        max_iter=200,
                                                                                        cv=10,
                                                                                        tol=1e-3,
                                                                                        solver='liblinear',
                                                                                        n_jobs=-1))]))


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

    # start loop
    for i in tqdm(range(len(participants))):
        eeg_data = []
        cat_labels = []
        sdt_clf = np.zeros((repeat_n, len(conditions), len(conditions), len(freq_vector)))
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
            print(test_i)

            eeg_train = eeg_data[train_i]
            cat_train = cat_labels[train_i]
            eeg_test = eeg_data[test_i]
            cat_test = cat_labels[test_i]
            ncat = len(np.unique(cat_labels))
            results_ = np.zeros((ncat, ncat, len(freq_vector)))
            for ii in range(len(freq_vector)):
                freq_ = freq_vector[ii]
                freq_train = eeg_train[:, :, freq_]
                freq_test = eeg_test[:, :, freq_]
                # fit classifier on train data with cross-validation
                internal_clf = clone(clf_model)
                fit_model = internal_clf.fit(freq_train, cat_train)
                prediction_ = fit_model.predict(freq_test)
                results_[:, :, i] = confusion_matrix(cat_test, prediction_)

            results_clf = ovo_clf_fpvs(train_i, test_i, clf_model, eeg_data, cat_labels, freq_vector[0:1])
            # returns d prime
            # results_clf[0] = cat x sdt measure x freq
            # results_clf[1] = cat x freq
            sdt_clf[rep_, :, :, :] = results_clf
            rep_ = rep_ + 1

