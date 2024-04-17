"""
====================================================================
Remote sensing: Change detection for sar images 
====================================================================

This example presents a change detection pipeline based on covariance matrices for sar data.
"""

# Author: Matthieu Verlynde and Ammar Mian

import os
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.pipeline import Pipeline

from numpy.lib.stride_tricks import sliding_window_view
from math import ceil 

from tqdm import trange

from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import ArrayLike

from joblib import Parallel, delayed
# import threading
import plotly.express as px
import sys


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Slinding window over an image of shape (n_rows, n_columns, n_features).

    Parameters
    ----------
    window_size : int
        Size of the window.

    overlap : int
        Overlap between windows. Default is 0.
    """

    def __init__(self, window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2,\
                'Overlap must be smaller or equal than int(window_size/2).'
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ArrayLike, y = None):
        return self

    def transform(self, X: ArrayLike):
        X = sliding_window_view(
                X,
                window_shape=(self.window_size, self.window_size),
                axis=(0, 1))
        if self.overlap is not None:
            if self.overlap > 0:
                X = X[::self.overlap, ::self.overlap]
        else:
            X = X[::self.window_size//2, ::self.window_size//2]
            self.overlap = self.window_size//2

        # Reshape to (n_pixels, n_images, n_features) with n_pixels=axis0*axis1 
        # n_images=T and n_features=axis4*axis5
        X = X.reshape((-1, X.shape[2], X.shape[3], X.shape[4]*X.shape[5]))
        return X
        
    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


class Covariance(BaseEstimator, TransformerMixin):
    """Slinding window over an image of shape (n_rows, n_columns, n_features).

    Parameters
    ----------
    window_size : int
        Size of the window.

    overlap : int
        Overlap between windows. Default is 0.
    """

    def __init__(self, n_jobs=1, verbose=0, method="function_self"):
        self.n_jobs = n_jobs
        self.verbose = verbose
        assert method in ["function_self", "function_pixel"], 'Method must be function_self or function_pixel.'
        self.method = method
        self.covar_all = None

    def fit(self, X: ArrayLike, y = None):
        if self.verbose>0:
            miniters = X.shape[0]//100 if X.shape[0]>100 else 1
            iter_pixel = trange(X.shape[0], miniters=miniters)
        else:
            iter_pixel = range(X.shape[0])

        if self.method == "function_self":
            p = X.shape[1]
            T = X.shape[2]
            self.covar_all=np.nan*np.ones((X.shape[0],T,p,p)).astype(complex)
            with Parallel(n_jobs=self.n_jobs) as parallel:
                result_covar = parallel(
                        delayed(self.get_covar)(X, i, T) for i in iter_pixel)
            for i,covar in enumerate(result_covar):
                self.covar_all[i] = covar

        elif self.method == "function_pixel":
            def compute_cov_pixel(x):
                p, n, T = x.shape
                covar = np.nan*np.ones((T,p,p)).astype(complex)
                for t in range(T):
                    covar[t] = np.cov(x[:, :, t])
                return covar
            with Parallel(n_jobs=self.n_jobs) as parallel:
                result_covar = parallel(
                        delayed(compute_cov_pixel)(X[i]) for i in iter_pixel)
                self.covar_all = np.stack(result_covar)

        return self

    def transform(self, X: ArrayLike):
        if self.covar_all is None:
            raise ValueError('The covariance matrices are not fitted.')
        return self.covar_all
    
    def get_covar(self, X: ArrayLike, i, T):
        covar = np.nan*np.ones((T,X.shape[1],X.shape[1])).astype(complex)
        for t in range(T):
            covar[t] = np.cov(X[i,:,t,:])
        return covar

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
        

class ChangeDetection(BaseEstimator, TransformerMixin):
    """Test for change detection using the covariance matrix of the SAR image."""

    def __init__(self, ENL: int, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.ENL = ENL
    
    def fit(self, X: ArrayLike, y=None):
        T = X.shape[1]
        p = X.shape[2]
        n = self.ENL
        self.parameters_predict = (T, p, n)

        self.lnq=np.nan*np.ones(X.shape[0])

        with Parallel(n_jobs=self.n_jobs) as parallel:
            result_lnq = parallel(delayed(self.get_lnq)(X, i, T, p, n) for i in range(X.shape[0]))
        for i,lnq in enumerate(result_lnq):
            self.lnq[i] = lnq

        return self
    
    def get_lnq(self, X: ArrayLike, i, T, p, n):
        Sigma_0 = np.zeros((p,p))
        result_denominator = 0
        for t in range(T):
            Sigma_t = X[i,t]
            Sigma_0 = Sigma_0 + Sigma_t
            result_denominator = result_denominator + np.log(np.abs(np.linalg.det(Sigma_t)))
        return n*(p*T*np.log(T) + result_denominator - T*np.log(np.abs(np.linalg.det(Sigma_0))))
    
    def transform(self, X: ArrayLike):
        return self.lnq

    def predict(self, X: ArrayLike):
        chi2 = scipy.stats.chi2.cdf
        T, p, n = self.parameters_predict
        f = (T-1)*(p**2)
        rho = 1 - (2*p**2-1)/(6*(T-1)*p)*(T/n-1/(n*T))
        omega_2 = (p**2)*(p**2-1)/(24*rho**2)*(T/(n**2)-1/(n*T)**2) -\
                (p**2)*(T-1)/4 * (1 - 1/rho)**2
        Z = -2*rho*self.lnq
        return chi2(Z, df=f) + omega_2*(chi2(Z, df=f+4) - chi2(Z, df=f))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    
    def fit_predict(self, X: ArrayLike, y=None):
        return self.fit(X).predict(X)



class LabelsToImage(BaseEstimator, TransformerMixin):
    """Predicted labels to image taking into account sliding windows.

    Parameters
    ----------
    height : int
        Height of the original image.

    width : int
        Width of the original image.

    window_size : int
        Size of the window.
    """

    def __init__(self, height: int, width: int,
                 window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2, \
                'Overlap must be smaller or equal than int(window_size/2).'
        self.height = height
        self.width = width
        self.overlap = overlap
        self.window_size = window_size

    def fit(self, X: ArrayLike, y=None):
        return self

    def transform(self, X: ArrayLike):
        # Compute reshape size thanks ot window-size before overlap
        height = self.height - self.window_size + 1
        width = self.width - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            height = ceil(height/self.overlap) 
            width = ceil(width/self.overlap)

        # Reshape to (height, weight)
        return X.reshape((height, width))

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)





if __name__ == "__main__":

    with open('python_process.pid', 'w', encoding='utf-8') as f:
        f.write(str(os.getpid()))

    # Parameters
    window_size = int(sys.argv[2])
    IN_DIR = "/home/verlyndem/Data/Selection"
    FILE_NAME = sys.argv[1] #"Scene_2.npy"
    n_jobs_cov = int(sys.argv[3])

    # IN_DIR = "/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection"
    # FILE_NAME = 'custom_test_image_n22500_T4_p3.npy'

    # Load data
    data = np.load(os.path.join(IN_DIR, FILE_NAME))

    # Pipelines definition
    pipeline = Pipeline([
        ('sliding_window', SlidingWindowVectorize(window_size=window_size)),
        ('covariances', Covariance(n_jobs=n_jobs_cov)),
        ('detector', ChangeDetection(ENL=window_size**2, n_jobs=n_jobs_cd))
        ],
        verbose=True)
        

    pipelines = [pipeline]
    pipelines_names = ['Change detection with sliding window and covariance matrix estimation : ' + FILE_NAME[:-4]]

    height, width = data.shape[:2]

    # Perform clustering
    results = {}
    for pipeline_name, pipeline in zip(pipelines_names, pipelines):
        print(f'Pipeline: {pipeline_name}')
        res_pipeline = pipeline.fit_transform(data)
        labels_pred = LabelsToImage(height, width, window_size).fit_transform(
                        res_pipeline
                )
        results[pipeline_name] = labels_pred

    # # Plot data
    # plot_value = np.sum(np.abs(data)**2, axis=2)
    # # plot_value = np.load(os.path.join(IN_DIR, FILE_NAME[:-4]+'_truth.npy'))
    # print(plot_value.shape)
    # figure = plt.figure(figsize=(10, 10))
    # plt.imshow(plot_value, aspect='auto')
    # plt.colorbar()
    # plt.title('Data')

    # # Plot results
    # for pipeline_name, labels_pred in results.items():
    #     figure = plt.figure(figsize=(10, 10))
    #     plt.imshow(labels_pred, aspect='auto', cmap='gray')
    #     plt.title(pipeline_name)
    #     plt.colorbar()
    # plt.show()


    # Save results

    # for pipeline_name, labels_pred in results.items():
    #     figure = px.imshow(labels_pred, aspect='auto', color_continuous_scale='gray', title=pipeline_name)
    # figure.write_html(FILE_NAME[:-4]+'_lnq.html', include_mathjax='cdn')
    
    # figure = px.imshow(plot_value, aspect='auto', title=FILE_NAME[:-4]+'_truth')
    # figure.write_html(FILE_NAME[:-4]+'_truth.html', include_mathjax='cdn')
    
    # figure = px.imshow(labels_pred>0.95, aspect='auto', color_continuous_scale='gray', title=FILE_NAME[:-4]+'_truth')
    # figure.write_html(FILE_NAME[:-4]+'_pvalue_threshold.html', include_mathjax='cdn')






