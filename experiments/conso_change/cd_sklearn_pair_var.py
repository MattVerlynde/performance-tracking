# -*- coding: utf-8 -*-
#
# This script is a python executable computing a change detection algorithm on time series of SAR images.
# Usage: python cd_sklearn_pair_var.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --image [PATH_TO_FOLDER_WITH_IMAGES] --window [WINDOW_SIZE] --cores [NUMBER_OF_CORES_USED] 
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 26 Jun 2024
# Version: 1.0.0


import os
import shutil
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

import argparse
import warnings

from helpers.multivariate_images_tool import sliding_windows_treatment_image_time_series_parallel, scale_and_shape_equality_robust_statistic, scale_and_shape_equality_statistic




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
        if overlap is not None:
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
        X = X.reshape((-1, X.shape[2], X.shape[3]*X.shape[4]))
        X = X.transpose((0, 2, 1))
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
                    covar[t] = np.cov(x[:, :, t], rowvar=True, bias=False)
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
            covar[t] = np.cov(X[i,:,t,:], rowvar=True, bias=False)
        return covar

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    
class LogDiffChangeDetection(BaseEstimator, TransformerMixin):
    """Test for change detection using the log difference between SAR images."""

    def __init__(self, threshold: float = 0.95, repeat: int = 1):
        self.threshold = threshold
        self.repeat = max(1,repeat)


    def fit(self, path: str, X=None, y=None):

        for r in range(self.repeat):
            list_images = os.listdir(path)

            image_t0 = np.load(os.path.join(path, list_images[0]))
            mean_diff = np.zeros(image_t0.shape[:2])
            T = len(list_images)

            def log_calc(image):
                return np.sum(np.log(image), axis= 2)

            image_t0 = log_calc(image_t0)

            for i in range(1,T):
                image_t1 = log_calc(np.load(os.path.join(path, list_images[i])))
                mean_diff += image_t0 - image_t1
                image_t0 = image_t1
            mean_diff = mean_diff/(T-1)
            self.result_diff = np.abs(mean_diff)
        
        return self
    
    def predict(self, X: ArrayLike):
        return self.result_diff > self.threshold
    
    def transform(self, X: ArrayLike):
        return self.result_diff
    
    def fit_predict(self, X: ArrayLike, y=None):
        return self.fit(X).predict(X)
    
    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    

class RobustChangeDetection(BaseEstimator, TransformerMixin):
    """Test for change detection using the covariance matrix of the SAR image."""

    def __init__(self, window_size: int, threshold: float = 0.95, n_jobs_cov: int = 1, tol: float = 0.001, iter_max: int = 20, scale: str = "linear"):
        self.threshold = threshold
        self.window_size = window_size
        self.ENL = window_size**2
        self.n_jobs_cov = n_jobs_cov
        self.tol = tol
        self.iter_max = iter_max
        self.scale = scale
    
    def fit(self, path: str, X=None, y=None):
        list_images = os.listdir(path)

        image = np.load(os.path.join(path, list_images[0]))
        n_r,n_c,p = image.shape[:3]
        multi = self.n_jobs_cov > 1
        T = len(list_images)
        # print(f"Multi: {multi}")

        function_args = (self.tol, self.iter_max, self.scale)

        image=image.reshape((n_r*n_c,p))
        for t in range(1,T):
            image = np.dstack((image, np.load(os.path.join(path, list_images[t])).reshape((n_r*n_c,p))))
        
        image=image.reshape((n_r,n_c,p,T))
        self.lambda_ = sliding_windows_treatment_image_time_series_parallel(image, 
                                                                            windows_mask=np.ones((self.window_size, self.window_size)), 
                                                                            function_to_compute=scale_and_shape_equality_robust_statistic, 
                                                                            function_args=function_args, 
                                                                            multi=multi, 
                                                                            number_of_threads_rows= self.n_jobs_cov//2, 
                                                                            number_of_threads_columns= self.n_jobs_cov//2, 
                                                                            progressbar=False)
        
        # image = image.transpose((3,0,1,2))

        # sliding_window = SlidingWindowVectorize(window_size=self.window_size)

        # image_windows = np.array([sliding_window.fit_transform(image[t]) for t in range(T)])
        # image_windows = image_windows.transpose((1,2,3,0))
        # image_windows = image_windows.transpose((0,2,1,3))

        # self.lambda_ = Parallel(n_jobs=self.n_jobs_cov)(
        #                 delayed(scale_and_shape_equality_robust_statistic)(x, args=(self.tol, self.iter_max, self.scale))
        #                 for x in image_windows
        #                 )
        # self.lambda_ = np.array(self.lambda_).reshape((n_r-2*(self.window_size//2),n_c-2*(self.window_size//2)))

        return self
    
    def predict(self, X: ArrayLike):
        return self.lambda_
    
    def transform(self, X: ArrayLike):
        return self.lambda_
    
    def fit_predict(self, X: ArrayLike, y=None):
        return self.fit(X).predict(X)
    
    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    
class GLRTChangeDetection(BaseEstimator, TransformerMixin):
    """Test for change detection using the covariance matrix of the SAR image."""

    def __init__(self, window_size: int, threshold: float = 0.95, n_jobs_cov: int = 1, tol: float = 0.001, iter_max: int = 20, scale: str = "linear"):
        self.threshold = threshold
        self.window_size = window_size
        self.ENL = window_size**2
        self.n_jobs_cov = n_jobs_cov
        self.tol = tol
        self.iter_max = iter_max
        self.scale = scale
    
    def fit(self, path: str, X=None, y=None):
        list_images = os.listdir(path)

        image = np.load(os.path.join(path, list_images[0]))
        n_r,n_c,p = image.shape[:3]
        multi = self.n_jobs_cov > 1
        T = len(list_images)
        # print(f"Multi: {multi}")

        function_args = (self.tol, self.iter_max, self.scale)

        image=image.reshape((n_r*n_c,p))
        for t in range(1,T):
            image = np.dstack((image, np.load(os.path.join(path, list_images[t])).reshape((n_r*n_c,p))))
        
        image=image.reshape((n_r,n_c,p,T))
        self.lambda_ = sliding_windows_treatment_image_time_series_parallel(image, 
                                                                            windows_mask=np.ones((self.window_size, self.window_size)), 
                                                                            function_to_compute=scale_and_shape_equality_statistic, 
                                                                            function_args=function_args, 
                                                                            multi=multi, 
                                                                            number_of_threads_rows= self.n_jobs_cov//2, 
                                                                            number_of_threads_columns= self.n_jobs_cov//2, 
                                                                            progressbar=False)
        
        # image = image.transpose((3,0,1,2))

        # sliding_window = SlidingWindowVectorize(window_size=self.window_size)

        # image_windows = np.array([sliding_window.fit_transform(image[t]) for t in range(T)])
        # image_windows = image_windows.transpose((1,2,3,0))
        # image_windows = image_windows.transpose((0,2,1,3))

        # self.lambda_ = Parallel(n_jobs=self.n_jobs_cov)(
        #                 delayed(scale_and_shape_equality_robust_statistic)(x, args=(self.tol, self.iter_max, self.scale))
        #                 for x in image_windows
        #                 )
        # self.lambda_ = np.array(self.lambda_).reshape((n_r-2*(self.window_size//2),n_c-2*(self.window_size//2)))

        return self
    
    def predict(self, X: ArrayLike):
        return self.lambda_
    
    def transform(self, X: ArrayLike):
        return self.lambda_
    
    def fit_predict(self, X: ArrayLike, y=None):
        return self.fit(X).predict(X)
    
    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    

class DataLoading(object):
    """Load the data from path."""

    def __init__(self, path: str):
        self.path = path
    
    def fit(self):
        return self
    
    def transform(self):
        return np.load(self.path)
    
    def fit_transform(self):
        return self.fit().transform()

class PairwiseRjTest(BaseEstimator, TransformerMixin):
    """Pairwise Rj test for change detection using the covariance matrix of the SAR image between two dates."""
    
    def __init__(self, window_size: int, n_jobs_cov: int = 10, return_count: bool = False, threshold: float = 0.95):
        self.window_size=window_size
        self.ENL = window_size**2
        self.threshold = threshold
        self.return_count = return_count
        self.n_jobs_cov = n_jobs_cov
        

    def fit(self, path: str, X=None, y=None):
        list_images = sorted(os.listdir(path))
        shape = np.load(os.path.join(path, list_images[0])).shape
        p = shape[2]
        n = self.ENL
        T = len(list_images)
        self.parameters_predict = (T, n, p)
        
        self.change_count = 0
        
        self.lnQ = None
        lnQ = 0

        sliding_window = SlidingWindowVectorize(window_size=self.window_size)

        for j in range(2,T+1):
            j_minus_1 = j-1

            # Load data
            
            image_j = DataLoading(os.path.join(path, list_images[j-1])).fit_transform()
            image_j_minus_1 = DataLoading(os.path.join(path, list_images[j_minus_1-1])).fit_transform()
            Sw_j = sliding_window.fit_transform(image_j)

            if j == 2:
                print("Computing covariances at date j=1")
                Sw_j_minus_1 = sliding_window.fit_transform(image_j_minus_1)
                Sw_j_minus_1 = np.transpose(Sw_j_minus_1, (0, 2, 1))
                # print(Sw_j_minus_1.shape)

                covar_j_minus_1 = Parallel(n_jobs=self.n_jobs_cov)(
                        delayed(np.cov)(x, rowvar=True, bias=False)
                        for x in Sw_j_minus_1
                        )
                # print(f"j = {j_minus_1}")
                # print(covar_j_minus_1[-1].shape)
                sum_covar_j_minus_1 = np.stack(covar_j_minus_1, axis=0)*n*T
            
            print(f"Adding cd between dates {j_minus_1} and {j}")       
            results = Parallel(n_jobs=self.n_jobs_cov)(
                delayed(self.rj_test)(x, j, sum_cov)
                        for x, sum_cov in zip(Sw_j, sum_covar_j_minus_1)
                )
            
            sum_covar_j_minus_1 = np.array([x[1] for x in results])
            lnQ += np.array([x[0] for x in results])
            print(lnQ[0])

        self.lnQ = lnQ

        return self
    
    def rj_test(self, X: ArrayLike, j, sum_covar_j_minus_1):
        T, n, p = self.parameters_predict
        covar_j = np.cov(X.T, rowvar=True, bias=False)*n*T
        # print(covar_j.shape)
        sum_covar_j = sum_covar_j_minus_1 + covar_j
        lnR = n*(p*(j*np.log(j) - (j-1)*np.log(j-1)) +\
            (j-1)*np.log(np.abs(np.linalg.det(sum_covar_j_minus_1))) +\
            np.log(np.abs(np.linalg.det(covar_j))) -\
            j*np.log(np.abs(np.linalg.det(sum_covar_j))))
        return lnR, sum_covar_j
    
    def transform(self, X: ArrayLike):
        return (-1)*self.lnQ
    
    def predict(self, X: ArrayLike):
        chi2 = scipy.stats.chi2.cdf
        T, n, p = self.parameters_predict
        f = (T-1)*(p**2)
        rho = 1 - ((2*p**2-1)/(6*(T-1)*p))*((T/n)-(1/(n*T)))
        omega_2 = (p**2)*((p**2-1)/(24*rho**2))*((T/(n**2))-(1/((n*T)**2))) -\
                (p**2)*((T-1)/4) * (1 - 1/rho)**2
        Z = -2*rho*self.lnQ
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

    def transform(self, X: ArrayLike, plot: bool = False):
        # Compute reshape size thanks ot window-size before overlap
        height = self.height - self.window_size + 1
        width = self.width - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            height = ceil(height/self.overlap) 
            width = ceil(width/self.overlap)

        # Reshape to (height, weight)
        X = X.reshape((height, width))

        if plot:
            self.plot(X)

        return X

    def fit_transform(self, X: ArrayLike, y=None, plot: bool = False):
        return self.fit(X).transform(X, plot=plot)
    
    def plot(self, X: ArrayLike):
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(X, aspect='auto', cmap='gray')
        plt.colorbar()
        plt.show()





if __name__ == "__main__":
    #Directory (the files correspond to npy files of the same scene for each date)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='data/Scene_1')
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--number_run", "-n", type=str, default="")
    parser.add_argument("--robust", type=int, required=0)
    args = parser.parse_args()

    print(args)

    DIR = args.image    
    window_size = int(args.window)
    n_jobs_cov = int(args.cores)
    
    # Pipelines definition
    if args.robust == 1:
        pipeline = Pipeline([
            ('robust_change_detection', RobustChangeDetection(window_size=window_size, threshold=0.95, n_jobs_cov = n_jobs_cov, tol=0.01, iter_max=2, scale="log"))
            ],
            verbose=False)
    elif args.robust == 2:
        pipeline = Pipeline([
            ('logdiff_change_detection', LogDiffChangeDetection(threshold=0.95, repeat=100))
            ],
            verbose=False)
    # elif args.robust == 0:
    #     pipeline = Pipeline([
    #         ('logdiff_change_detection', GLRTChangeDetection(threshold=0.95, repeat=100))
    #         ],
    #         verbose=False)
    else:
        pipeline = Pipeline([
            ('rj_test', PairwiseRjTest(window_size=window_size, n_jobs_cov=n_jobs_cov, return_count=False, threshold=0.95))
            ],
            verbose=False)

    height, width =  np.load(os.path.join(DIR, os.listdir(DIR)[0])).shape[:2]

    from codecarbon import OfflineEmissionsTracker
    
    DIR_CARBON = os.path.join(args.storage_path,"codecarbon")
    DIR_OUTPUT = os.path.join(args.storage_path,"output")
    os.makedirs(DIR_CARBON, exist_ok=True)
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=DIR_CARBON, output_file=f"emissions{args.number_run}.csv")
    tracker.start()

    res_pipeline = pipeline.fit_transform(DIR)
    labels_pred = LabelsToImage(height, width, window_size).fit_transform(
        res_pipeline
        )
    np.save(os.path.join(DIR_OUTPUT,args.number_run+'.npy'), labels_pred)
    
    tracker.stop()






