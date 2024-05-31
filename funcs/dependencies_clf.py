import os
import pandas as pd
import scipy.io as sio


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
    # header['chanlocs']
    return epochs, xstart, xstep, xlen


# Function to map Bayes factor to qualitative evidence log
def interpret_bayes_factor_log(bf):
    bf = float(bf)
    if bf < -2:
        return "Decisive (null)"
    elif -2 <= bf < -1.5:
        return "Strong 2 (null)"
    elif -1.5 <= bf < -1:
        return "Strong (null)"
    elif -1 <= bf < -0.5:
        return "Substantial (null)"
    elif -0.5 <= bf < 0.5:
        return "Barely worth mentioning"
    elif 0.5 <= bf < 1:
        return "Substantial"
    elif 1 <= bf < 1.5:
        return "Strong"
    elif 1.5 <= bf < 2:
        return "Strong 2"
    else:  # bf >= 100
        return "Decisive"




