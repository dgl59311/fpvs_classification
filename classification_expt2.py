# Classification analysis
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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone
from funcs.dependencies_clf import data_check, find_headers, experiment_info
import matplotlib
import matplotlib.pyplot as plt

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
                            ('clf_model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3))])

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]

# Select which experiment to run
expt = 2
print('Runnin code for experiment: ', expt)
conditions, clf_files_local, _fname_ = experiment_info(expt)
files_ = data_check(conditions, clf_files_local)
files_ = files_[files_.index.isin(demog_file.index)]
participants = files_.index

# start loop
for i in tqdm(range(len(participants))):
    eeg_data = []
    cat_labels = []
    f1_clf = np.zeros((repeat_n, len(freq_vector)))
    acc_clf = np.zeros((repeat_n, len(freq_vector)))
    sdt_clf = np.zeros((repeat_n, 4, len(freq_vector)))
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
    # repeated train-test split for unbiased estimates of accuracy with small sample sizes
    # sample 60% of the trials, train the model and test on the remaining 40% ... repeat this 100 times
    rep_ = 0
    for train_i, test_i in tqdm(random_sp.split(eeg_data, cat_labels), total=random_sp.get_n_splits()):
        eeg_train = eeg_data[train_i]
        cat_train = cat_labels[train_i]
        eeg_test = eeg_data[test_i]
        cat_test = cat_labels[test_i]
        for freq_ in range(len(freq_vector)):
            freq_train = eeg_train[:, :, freq_]
            freq_test = eeg_test[:, :, freq_]
            # fit classifier on train data with cross-validation
            internal_clf = clone(clf_model)
            fit_model = internal_clf.fit(freq_train, cat_train)
            prediction_ = fit_model.predict(freq_test)
            tn, fp, fn, tp = confusion_matrix(cat_test, prediction_).ravel()
            sdt_clf[rep_, :, freq_] = [tp, fn, fn, tn]
            f1_clf[rep_, freq_] = f1_score(cat_test, prediction_)
            acc_clf[rep_, freq_] = accuracy_score(cat_test, prediction_)
        rep_ = rep_ + 1
    # save data
    np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + '_f1_expt_2.npy'), f1_clf)
    np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + '_acc_expt_2.npy'), acc_clf)
    np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + '_expt_2_tp_fp_fn_tn.npy'), sdt_clf)

