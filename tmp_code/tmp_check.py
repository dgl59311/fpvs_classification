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
from sklearn.datasets import make_classification
from sklearn.base import clone


X, y = make_classification(
    n_samples=500,
    n_features=64,
    n_informative=3,
    n_classes=4,
    random_state=999
)

clf_model_mc = Pipeline(steps=[('scale', StandardScaler()),
                               ('pca', PCA(n_components=20)),
                               ('clf_model', OneVsOneClassifier(LogisticRegressionCV(Cs=np.logspace(-5, 5, 30),
                                                                                     penalty='l1',
                                                                                     max_iter=200,
                                                                                     cv=5,
                                                                                     tol=1e-3,
                                                                                     solver='liblinear',
                                                                                     n_jobs=-1)))])

random_sp = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=234)
for train_i, test_i in random_sp.split(X, y):
    internal_clf = clone(clf_model_mc)
    fit_model = internal_clf.fit(X[train_i, :], y[train_i])
    prediction_ = fit_model.predict(X[test_i, :])

# Access the coefficients of each binary classifier within OneVsOneClassifier
binary_classifiers = fit_model.named_steps['clf_model'].estimators_
selected_features = []

for i, classifier in enumerate(binary_classifiers):
    if hasattr(classifier, 'coef_'):
        # Retrieve the non-zero coefficients for the i-th binary classifier
        non_zero_indices = np.where(classifier.coef_[0] != 0)[0]
        selected_features.append((f"Classifier {i}", non_zero_indices))

# selected_features now contains a list of tuples where each tuple
# corresponds to a binary classifier and the selected feature indices
for classifier_num, feature_indices in selected_features:
    print(f"Selected Features for {classifier_num}: {feature_indices}")