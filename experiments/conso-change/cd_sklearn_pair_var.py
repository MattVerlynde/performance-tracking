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

class RobustChangeDetection(BaseEstimator, TransformerMixin):
    """Test for change detection using the covariance matrix of the SAR image."""

    def __init__(self, window_size: int, n_jobs_cov: int = 10, return_count: bool = False, threshold: float = 0.95):
        self.window_size=window_size
        self.ENL = window_size**2
        self.threshold = threshold
        self.return_count = return_count
        self.n_jobs_cov = n_jobs_cov

    def q(Sigma, X):
        return X.conj().T@np.linalg.inv(Sigma)@X
        

    def fit(self, path: str, X=None, y=None):
        list_images = os.listdir(path)
        sum_covar_j_minus_1 = 0
        sum_covar_j = 0
        shape = np.load(os.path.join(path, list_images[0])).shape
        p = shape[2]
        n = self.ENL
        T = len(list_images)
        self.parameters_predict = (T, p, n)
        
        self.change_count = 0
        
        self.lnq = 0

        # Pipelines definition
        pipeline = Pipeline([
            ('sliding_window', SlidingWindowVectorize(window_size=self.window_size)),
            ('covariances', Covariance(n_jobs=self.n_jobs_cov))
            ],
            verbose=False)
        
        image = DataLoading(os.path.join(path, list_images[0])).fit_transform().reshape((shape[0], shape[1], shape[2], 1))
        covar_j = pipeline.fit_transform(image).reshape((-1, p, p))

        for j in range(2,T+1):
            j_minus_1 = j-1

            # Load data
            image = DataLoading(os.path.join(path, list_images[j-1])).fit_transform().reshape((shape[0], shape[1], shape[2], 1))
            
            covar_j_minus_1 = covar_j
            covar_j = pipeline.fit_transform(image).reshape((-1, p, p))
            sum_covar_j_minus_1 = sum_covar_j_minus_1 + covar_j_minus_1
            sum_covar_j = sum_covar_j_minus_1 + covar_j
            

            self.lnRj = n*(
                p*(j*np.log(j) - j_minus_1*np.log(j_minus_1)) +
                  j_minus_1*np.log(np.abs(np.linalg.det(sum_covar_j_minus_1))) +
                  np.log(np.abs(np.linalg.det(covar_j))) - 
                  j*np.log(np.abs(np.linalg.det(sum_covar_j)))
                  ) 
            
            if self.return_count:
                f = p**2
                rhoj = 1 - (2*p**2 - 1)/(6*p*n) * (1 + 1/(j*j_minus_1))
                omega_2j = -p**2/4 * (1 - 1/rhoj)**2 + 1/(24*n**2)*p**2*(p**2-1)*(1 + (2*j-1)/(j**2*j_minus_1**2))*1/rhoj**2
                
                chi2 = scipy.stats.chi2.cdf
                Z = -2*rhoj*self.lnRj
                
                pvalue = chi2(Z, df=f) + omega_2j * (chi2(Z, df=f+4) - chi2(Z, df=f))
                
                self.change_count = self.change_count + (pvalue > self.threshold)

            self.lnq += self.lnRj

            # labels = LabelsToImage(shape[0], shape[1], window_size).fit_transform(
            #             pvalue,
            #             plot=plot
            #     )
 
        return self
    
    def transform(self, X: ArrayLike):
        return self.lambda_
    
    def predict(self, X: ArrayLike):
        # if self.return_count:
        #     return self.change_count
        # chi2 = scipy.stats.chi2.cdf
        # T, p, n = self.parameters_predict
        # f = (T-1)*(p**2)
        # rho = 1 - (2*p**2-1)/(6*(T-1)*p)*(T/n-1/(n*T))
        # omega_2 = (p**2)*(p**2-1)/(24*rho**2)*(T/(n**2)-1/(n*T)**2) -\
        #         (p**2)*(T-1)/4 * (1 - 1/rho)**2
        # Z = -2*rho*self.lnq
        # return chi2(Z, df=f) + omega_2*(chi2(Z, df=f+4) - chi2(Z, df=f))
        return self.lambda_
    
    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
    
    def fit_predict(self, X: ArrayLike, y=None):
        return self.fit(X).predict(X)
        

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
            Sigma_t = n*X[i,t]
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
        list_images = os.listdir(path)
        sum_covar_j_minus_1 = 0
        sum_covar_j = 0
        shape = np.load(os.path.join(path, list_images[0])).shape
        p = shape[2]
        n = self.ENL
        T = len(list_images)
        self.parameters_predict = (T, p, n)
        
        self.change_count = 0
        
        self.lnq = 0

        # Pipelines definition
        pipeline = Pipeline([
            ('sliding_window', SlidingWindowVectorize(window_size=self.window_size)),
            ('covariances', Covariance(n_jobs=self.n_jobs_cov))
            ],
            verbose=False)
        
        image = DataLoading(os.path.join(path, list_images[0])).fit_transform().reshape((shape[0], shape[1], shape[2], 1))
        covar_j = pipeline.fit_transform(image).reshape((-1, p, p))

        for j in range(2,T+1):
            j_minus_1 = j-1

            # Load data
            image = DataLoading(os.path.join(path, list_images[j-1])).fit_transform().reshape((shape[0], shape[1], shape[2], 1))
            
            covar_j_minus_1 = covar_j
            covar_j = pipeline.fit_transform(image).reshape((-1, p, p))
            sum_covar_j_minus_1 = sum_covar_j_minus_1 + covar_j_minus_1
            sum_covar_j = sum_covar_j_minus_1 + covar_j
            

            self.lnRj = n*(
                p*(j*np.log(j) - j_minus_1*np.log(j_minus_1)) +
                  j_minus_1*np.log(np.abs(np.linalg.det(sum_covar_j_minus_1))) +
                  np.log(np.abs(np.linalg.det(covar_j))) - 
                  j*np.log(np.abs(np.linalg.det(sum_covar_j)))
                  ) 
            
            if self.return_count:
                f = p**2
                rhoj = 1 - (2*p**2 - 1)/(6*p*n) * (1 + 1/(j*j_minus_1))
                omega_2j = -p**2/4 * (1 - 1/rhoj)**2 + 1/(24*n**2)*p**2*(p**2-1)*(1 + (2*j-1)/(j**2*j_minus_1**2))*1/rhoj**2
                
                chi2 = scipy.stats.chi2.cdf
                Z = -2*rhoj*self.lnRj
                
                pvalue = chi2(Z, df=f) + omega_2j * (chi2(Z, df=f+4) - chi2(Z, df=f))
                
                self.change_count = self.change_count + (pvalue > self.threshold)

            self.lnq += self.lnRj

            # labels = LabelsToImage(shape[0], shape[1], window_size).fit_transform(
            #             pvalue,
            #             plot=plot
            #     )
 
        return self
    
    def transform(self, X: ArrayLike):
        return self.lnq
    
    def predict(self, X: ArrayLike):
        if self.return_count:
            return self.change_count
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
    parser.add_argument("--image", type=str, default='/home/verlyndem/Data/Selection/500x500x4')
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--number_run", "-n", type=str, default="")
    args = parser.parse_args()

    DIR = args.image # "Scene_2"
    # HOME_DIR = "/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection/"
    # DIR = 'custom_test_image_n22500_T4_p3_2'
    
    window_size = int(args.window) #11
    n_jobs_cov = int(args.cores)
    
    # Pipelines definition
    pipeline = Pipeline([
        ('rj_test', PairwiseRjTest(window_size=window_size, n_jobs_cov=n_jobs_cov,return_count=False, threshold=0.95))
        ],
        verbose=False)
    # pipeline = Pipeline([
    #     ('robust_change_detection', RobustChangeDetection(window_size=window_size, n_jobs_cov=n_jobs_cov, tol=0.0001, iter_max=20))
    #     ],
    #     verbose=False)
        
    pipelines = [pipeline]
    name = 'pairwise_change_detection'
    pipelines_names = [name]

    height, width =  np.load(os.path.join(DIR, os.listdir(DIR)[0])).shape[:2]

    # Perform clustering
    results = {}

    from codecarbon import OfflineEmissionsTracker
    
    DIR_CARBON = os.path.join(args.storage_path,"codecarbon")
    os.makedirs(DIR_CARBON, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=DIR_CARBON, output_file=f"emissions{args.number_run}.csv")
    tracker.start()

    for pipeline_name, pipeline in zip(pipelines_names, pipelines):
        res_pipeline = pipeline.fit_predict(DIR)
        labels_pred = LabelsToImage(height, width, window_size).fit_transform(
                        res_pipeline
                )
        results[pipeline_name] = labels_pred

        np.save(os.path.join(args.storage_path,"output/"+pipeline_name)+args.number_run+'.npy', labels_pred)
    
    tracker.stop()






