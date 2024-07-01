import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from Mondrian_RF.Mondrian_forest import MondrianForestRegressor, MondrianForestTransformer
from others.sir import SlicedInverseRegression
from others.save import SlicedAverageVarianceEstimation
from others.kernel_regression import fit_kernel_smoother_silverman
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from dataset import (
    load_abalone,
    load_kin8nm,
    load_openml,
    load_diabetes
)


DATASETS = {
    'abalone': load_abalone,
    'diabetes': load_diabetes,
    'autoprice': partial(load_openml, name='autoPrice'),
    'mu284': partial(load_openml, name='mu284'),
    'bank8FM': partial(load_openml, name='bank8FM'),
    'kin8nm' : load_kin8nm,
}


OUT_DIR = 'real_data_results'


def fit_RF(X_train, y_train, X_test, y_test, n_estimators, parameters):
    forest = RandomForestRegressor(n_estimators = n_estimators, random_state=123)
    clf = GridSearchCV(forest, parameters, n_jobs=-1).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    err = np.mean((y_pred - y_test)**2)
    print(err)
    return err


def benchmark(dataset, n_resamples=15, n_splits=10):
    X, y = DATASETS[dataset]()
    n_samples, n_features = X.shape
    n_estimators = 10

    rf_parameters = {
        'min_samples_leaf': [1, 5],
        'max_features': [2, 4, 6, 1/3., 'sqrt', None]
    }

    mf_parameters = {
        'lifetime': [1, 2, 3, 4, 5]
    }

    mft_parameters = {
        'step_size': [0.1, 0.2, 0.5],
        'iteration': [1, 2]
    }

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for resample_id in tqdm(range(n_resamples)):
        cv = KFold(n_splits=n_splits, shuffle=True,
                   random_state=resample_id * 42)
        results = {
            'mean': np.zeros(n_splits),
            'kernel_reg_raw': np.zeros(n_splits),
            'kernel_reg_sir': np.zeros(n_splits),
            'kernel_reg_save': np.zeros(n_splits),
            'rf': np.zeros(n_splits),
            'sir_rf': np.zeros(n_splits),
            'save_rf': np.zeros(n_splits),
            'mf': np.zeros(n_splits),
            'mft': np.zeros(n_splits)
        }

        for k, (train, test) in enumerate(cv.split(X)):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            print("Mean Only")
            err = np.mean((y_test - np.mean(y_train))**2)
            results['mean'][k] = err
            print(err)

            if X.shape[1] < 6:
                rf_parameters['max_features'] = [2, 4, 1/3., 'sqrt', None]
            elif X.shape[1] < 4:
                rf_parameters['max_features'] = [2, 1/3., 'sqrt', None]
            
            print("Random Forest")
            results['rf'][k] = fit_RF(X_train, y_train, X_test, y_test, n_estimators, rf_parameters)

            print("SIR Random Forest")
            sir = SlicedInverseRegression(n_directions=None)
            sir.fit(X_train, y_train)
            X_train_sir = sir.transform(X_train)
            X_test_sir = sir.transform(X_test)
            results['sir_rf'][k] = fit_RF(X_train_sir, y_train, X_test_sir, y_test, n_estimators, rf_parameters)
            
            print("SAVE Random Forest")
            save = SlicedAverageVarianceEstimation(n_directions=None)
            save.fit(X_train, y_train)
            X_train_save = save.transform(X_train)
            X_test_save = save.transform(X_test)
            results['save_rf'][k] = fit_RF(X_train_save, y_train, X_test_save, y_test, n_estimators, rf_parameters)


            for feature_type in ['raw', 'sir', 'save']:
                print(f"{feature_type} Kernel Regression")
                ksmooth = fit_kernel_smoother_silverman(
                    X_train, y_train, feature_type=feature_type)
                y_pred = ksmooth.predict(X_test)
                err = np.mean((y_pred - y_test)**2)
                results['kernel_reg_{}'.format(feature_type)][k] = err
                print(err)

            # Mondrian Forest
            print("Mondrian Forest")
            
            mf = MondrianForestRegressor(n_estimators=n_estimators, random_state=123)
            clf = GridSearchCV(mf, mf_parameters, n_jobs=-1).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            err = np.mean((y_pred - y_test)**2)
            results['mf'][k] = err
            print(clf.best_params_)
            print(err)

            mft = MondrianForestTransformer(mf = clf.best_estimator_)
            clf = GridSearchCV(mft, mft_parameters, n_jobs=-1).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            err = np.mean((y_pred - y_test)**2)
            results['mft'][k] = err
            print(clf.best_params_)
            print(err)

        results = pd.DataFrame(results)
        results['fold'] = np.arange(n_splits)

        output_name = os.path.join(OUT_DIR, "{}_{}n_{}p_{}k_{}r_{}.csv".format(
            dataset, n_samples, n_features, n_splits, n_resamples, resample_id))
        results.to_csv(output_name, index=False)


if __name__ == '__main__':
    benchmark('abalone')
    benchmark('diabetes')
    benchmark('autoprice')
    benchmark('mu284')
    benchmark('bank8FM')
    benchmark('kin8nm')