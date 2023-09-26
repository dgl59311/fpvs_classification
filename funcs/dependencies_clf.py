import os
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.base import clone


def experiment_info(id_experiment):
    # get info from each experiment
    if id_experiment == 1:
        conditions = ['03Hz', '06Hz', '09Hz', '12Hz']
        dir_files = "C:/Users/gordillo/Desktop/local_fpvs/Experiment 1"
        id_files = '_experiment_1'

    elif id_experiment == 2:
        conditions = ['house', 'face']
        dir_files = "C:/Users/gordillo/Desktop/local_fpvs/Experiment 2"
        id_files = '_experiment_2'
    else:
        conditions = ['10Hz050', '10Hz100', '20Hz100']
        dir_files = "C:/Users/gordillo/Desktop/local_fpvs/Experiment 3"
        id_files = '_experiment_3'
    return conditions, dir_files, id_files


def data_check(conditions, folder):
    # the output is a dataframe with the name of the participant and the location of the data for each condition
    for i in range(len(conditions)):
        list_files = list(filter(lambda s: '.mat' in s and conditions[i] in s, os.listdir(folder)))
        common_prefix = os.path.commonprefix(list_files)
        common_suffix = os.path.commonprefix([s[::-1] for s in list_files])[::-1]
        result_list = [s.replace(common_prefix, '').replace(common_suffix, '') for s in list_files]
        if i == 0:
            df_results = pd.DataFrame(index=result_list, data=list_files, columns=[conditions[i]])
        else:
            tmp_df = pd.DataFrame(index=result_list, data=list_files, columns=[conditions[i]])
            df_results = pd.concat([df_results, tmp_df], axis=1, sort=False)
    return df_results


def find_headers(matfilepath):
    # read info file from letswave
    header = sio.loadmat(matfilepath.replace(".mat", ".lw6"))['header']
    # number of epochs available
    epochs = header['datasize'].item()[0][0]
    xstart = header['xstart'].item()[0][0]
    xstep = header['xstep'].item()[0][0]
    xlen = header['datasize'].item()[0][5]
    return epochs, xstart, xstep, xlen


def sdt_multinomial(confusion_mat):

    num_classes = confusion_mat.shape[0]
    sdt_values = np.zeros((num_classes, 4))

    for i in range(num_classes):

        tp = confusion_mat[i, i]
        fp = np.sum(confusion_mat[:, i]) - tp
        fn = np.sum(confusion_mat[i, :]) - tp
        tn = np.sum(confusion_mat) - (tp + fp + fn)

        sdt_values[i, 0] = tp
        sdt_values[i, 1] = fp
        sdt_values[i, 2] = fn
        sdt_values[i, 3] = tn

    return sdt_values


def dprime_clf(tp, fp, fn, tn):

    # loglinear dprime
    H = (tp + 0.5) / (tp + fn + 1)
    FA = (fp + 0.5) / (fp + tn + 1)

    Z_H = stats.norm.ppf(H)
    Z_FA = stats.norm.ppf(FA)
    d_prime = Z_H - Z_FA

    return d_prime


def clf_fpvs(train_index, test_index, model, data_x, data_y, freq):
    eeg_train = data_x[train_index]
    cat_train = data_y[train_index]
    eeg_test = data_x[test_index]
    cat_test = data_y[test_index]
    n_cat = len(np.unique(data_y))
    results_ = np.zeros((n_cat, 4, len(freq)))
    d_primes = np.zeros((n_cat, len(freq)))
    for i in range(len(freq)):
        freq_ = freq[i]
        freq_train = eeg_train[:, :, freq_]
        freq_test = eeg_test[:, :, freq_]
        # fit classifier on train data with cross-validation
        internal_clf = clone(model)
        fit_model = internal_clf.fit(freq_train, cat_train)
        prediction_ = fit_model.predict(freq_test)
        if n_cat > 2:
            conf_mat = confusion_matrix(cat_test, prediction_)
            # gives a matrix of category x sdt measure (TP, FP, FN, TN)
            sdt_ = sdt_multinomial(conf_mat)
            results_[:, :, i] = sdt_
            for j in range(n_cat):
                d_primes[j, i] = dprime_clf(sdt_[j, 0], sdt_[j, 1], sdt_[j, 2], sdt_[j, 3])
        else:
            tn, fp, fn, tp = confusion_matrix(cat_test, prediction_).ravel()
            d_primes[0, i] = dprime_clf(tp, fp, fn, tn)

    return results_, d_primes
