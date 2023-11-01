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
from funcs.dependencies_clf import data_check, find_headers, bin_clf_fpvs, experiment_info, clf_id
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
clf_model = Pipeline(steps=[('scale', StandardScaler()),
                            ('pca', PCA(n_components=20)),
                            ('clf_model', LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                               penalty='l1',
                                                               max_iter=200,
                                                               cv=10,
                                                               tol=1e-3,
                                                               solver='liblinear',
                                                               n_jobs=-1))])

# frequencies to analyze
freq_vector = np.arange(0, 257, 1)
freq_vector = freq_vector[freq_vector < 48]

# read demog file
demog_file = pd.read_csv('SR_Testing_FPVS.csv', index_col=0)
demog_file = demog_file[(demog_file['Label'] == 'Control') | (demog_file['Label'] == 'SR1')]

# Select which experiment to run
# 1: model for Category selectivity
# 2: model for Duty Cycle - 10Hz050 vs 10Hz100
# 3: model for Duty Cycle - 10Hz100 vs 20Hz100

for model_ in range(5, 6):
    # returns the experiment id and file names of the 5 classifiers
    # see function in dependences_clf.py
    expt, id_files = clf_id(model_)
    print(id_files)

    print('Runnin code for experiment: ', expt)
    conditions, clf_files_local, _fname_ = experiment_info(expt)
    files_ = data_check(conditions, clf_files_local)
    files_ = files_[files_.index.isin(demog_file.index)]
    participants = files_.index

    # start loop
    for i in tqdm(range(len(participants))):
        eeg_data = []
        cat_labels = []
        d_clf = np.zeros((repeat_n, len(freq_vector)))
        sdt_clf = np.zeros((repeat_n, 4, len(freq_vector)))
        d_clf_rlp = np.zeros((repeat_n, len(freq_vector)))
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

        if model_ == 0:
            # this model corresponds to 6Hz vs 3Hz
            eeg_data = eeg_data[(cat_labels != 2) & (cat_labels != 3), :, :]
            cat_labels = cat_labels[(cat_labels != 2) & (cat_labels != 3)]
        elif model_ == 1:
            # this model corresponds to 6Hz vs 9Hz
            eeg_data = eeg_data[(cat_labels != 0) & (cat_labels != 3), :, :]
            cat_labels = cat_labels[(cat_labels != 0) & (cat_labels != 3)]
        elif model_ == 2:
            # this model corresponds to 6Hz vs 12Hz
            eeg_data = eeg_data[(cat_labels != 0) & (cat_labels != 2), :, :]
            cat_labels = cat_labels[(cat_labels != 0) & (cat_labels != 2)]
        elif model_ == 4:
            # this model corresponds to 10Hz050 vs 10Hz100
            eeg_data = eeg_data[cat_labels != 2, :, :]
            cat_labels = cat_labels[cat_labels != 2]
        elif model_ == 5:
            # this model corresponds to 10Hz100 vs 20Hz100
            eeg_data = eeg_data[cat_labels != 0, :, :]
            cat_labels = cat_labels[cat_labels != 0]

        # for "true" data
        rep_ = 0
        for train_i, test_i in random_sp.split(eeg_data, cat_labels):
            results_clf = bin_clf_fpvs(train_i, test_i, clf_model, eeg_data, cat_labels, freq_vector)
            # returns d prime
            # results_clf[0] = cat x sdt measure x freq
            # results_clf[1] = cat x freq
            sdt_clf[rep_, :, :] = results_clf[0]
            d_clf[rep_, :] = results_clf[1]
            rep_ = rep_ + 1
        # save data
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + '.npy'), d_clf)
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + 'tp_fp_fn_tn.npy'), sdt_clf)

        # for random_label permutations
        rep_rlp = 0
        for k in range(repeat_n):
            indices = np.random.permutation(len(cat_labels))
            rlp_labels = cat_labels[indices]
            for train_ri, test_ri in random_sp_rlp.split(eeg_data, rlp_labels):
                results_clf_rlp = bin_clf_fpvs(train_ri, test_ri, clf_model, eeg_data, rlp_labels, freq_vector)
                d_clf_rlp[rep_rlp, :] = results_clf_rlp[1]
            rep_rlp = rep_rlp + 1

        # save data
        np.save(os.path.join(main_dir, 'results', 'classification_data', participants[i] + id_files + 'rlp.npy'), d_clf_rlp)

