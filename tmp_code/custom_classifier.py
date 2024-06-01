import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelEncoder


class OneVsOneClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=None, random_state=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.estimators_ = []

        random_state = check_random_state(self.random_state)

        combinations = list(self._yield_combinations(self.classes_))
        self.pairwise_indices_list = []

        for classes in combinations:
            indices = np.where((y == classes[0]) | (y == classes[1]))[0]
            self.pairwise_indices_list.append(indices)

            if len(indices) > 0:
                X_pairwise = X[indices]
                y_pairwise = y[indices]

                estimator = self._make_estimator()
                estimator.fit(X_pairwise, y_pairwise)
                self.estimators_.append(estimator)

        return self

    def _yield_combinations(self, classes):
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                yield (classes[i], classes[j])

    def _make_estimator(self):
        return clone(self.estimator)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = []

        for indices, estimator in zip(self.pairwise_indices_list, self.estimators_):
            if len(indices) > 0:
                predictions.append(estimator.predict(X[indices]))
            else:
                # If the training set for this pair was empty, predict a constant value
                predictions.append(np.full(X.shape[0], fill_value=self.classes_[0]))

        # Calculate the average decision values for each class
        decision_values = np.vstack([estimator.decision_function(X) for estimator in self.estimators_])
        avg_decision_values = np.mean(decision_values, axis=0)

        # Break ties by choosing the class with the highest average decision value
        final_predictions = np.argmax(avg_decision_values, axis=1)

        return final_predictions
