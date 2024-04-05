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

import plotly.express as px

import sklearn.covariance as skcov

from pyriemann.estimation import Covariances as RiemannCovariances

def covariance_matrix(X, mode = "empirical", alpha=0.5):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * X = an array
                * mode = a string with the type of covariance estimation algorithm
            Outputs:
                * the covariance matrix """
    
    assert mode in ["empirical", "sample", "ledoit-wolf", "oas"], 'Mode must be empirical, sample, ledoit-wolf, oas, shrinkage or graphical-lasso.'
    
    if mode == "sample":
        return np.corrcoef(X)
    
    elif mode == "ledoit-wolf":
        # cov, _ = skcov.ledoit_wolf(X)
        estimators = ["scm", "lwf", "oas", "mcd", "hub"]
        cov = RiemannCovariances(estimator="lwf").transform(X)[0]
        return cov
    
    elif mode == "oas":
        # cov, _ = skcov.oas(X)
        cov = RiemannCovariances(estimator="oas").transform(X)[0]
        return cov
    
    elif mode == "shrinkage":
        cov, _ = skcov.shrunk_covariance(X)
        return cov
    
    elif mode == "graphical-lasso":
        cov = skcov.empirical_covariance(X)
        cov, _ = skcov.graphical_lasso(cov,alpha)
        return cov
    
    return np.cov(X)

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

    def __init__(self, n_jobs=1, verbose=0, method="function_self", mode_covar="empirical", alpha=0.5):
        self.n_jobs = n_jobs
        self.verbose = verbose
        assert method in ["function_self", "function_pixel"], 'Method must be function_self or function_pixel.'
        self.method = method
        self.covar_all = None
        self.mode_covar = mode_covar
        self.alpha = alpha

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
                        delayed(self.get_covar)(X, i, T, self.mode_covar, self.alpha) for i in iter_pixel)
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
    
    def get_covar(self, X: ArrayLike, i, T, mode="empirical", alpha=0.5):
        covar = np.nan*np.ones((T,X.shape[1],X.shape[1])).astype(complex)
        for t in range(T):
            covar[t] = covariance_matrix(X[i,:,t,:], mode, alpha)
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
    
    def __init__(self, window_size: int, pvalue_rj: bool = False, threshold: float = 0.95, mode_covar: str = "empirical", alpha : float = 0.5):
        self.window_size=window_size
        self.ENL = window_size**2
        self.threshold = threshold
        self.pvalue_rj = pvalue_rj
        self.mode_covar = mode_covar
        self.alpha = alpha


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
            ('covariances', Covariance(n_jobs=10, mode_covar=self.mode_covar, alpha=self.alpha))
            ],
            verbose=True)
        
        image = DataLoading(os.path.join(path, list_images[0])).fit_transform().reshape((shape[0], shape[1], shape[2], 1))
        covar_j = pipeline.fit_transform(image).reshape((-1, p, p))

        for j in trange(2,T+1):
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
            
            if self.pvalue_rj:
                f = p**2
                rhoj = 1 - (2*p**2 - 1)/(6*p*n) * (1 + 1/(j*j_minus_1))
                omega_2j = -p**2/4 * (1 - 1/rhoj)**2 + 1/(24*n**2)*p**2*(p**2-1)*(1 + (2*j-1)/(j**2*j_minus_1**2))*1/rhoj**2
                
                chi2 = scipy.stats.chi2.cdf
                Z = -2*rhoj*self.lnRj
                
                pvalue = chi2(Z, df=f) + omega_2j * (chi2(Z, df=f+4) - chi2(Z, df=f))
                

                plt.imshow(LabelsToImage(height=shape[0], width=shape[1], window_size=self.window_size).fit_transform(
                    pvalue>0.99), aspect='auto', cmap='gray')
                plt.title(j)
                plt.colorbar()
                plt.show()

                figure = px.imshow(LabelsToImage(height=shape[0], width=shape[1], window_size=self.window_size).fit_transform(
                    pvalue>0.99), aspect='auto', color_continuous_scale='gray', title=str(j))
                figure.write_html(str(j)+'_pvalue_threshold.html', include_mathjax='cdn')
                
                figure = px.imshow(LabelsToImage(height=shape[0], width=shape[1], window_size=self.window_size).fit_transform(
                    pvalue), aspect='auto', color_continuous_scale='gray', title=str(j))
                figure.write_html(str(j)+'_pvalue.html', include_mathjax='cdn')

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
        if self.pvalue_rj:
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
    # HOME_DIR = "/home/verlyndem/Data/Selection/"
    # DIR = "Scene_3"
    HOME_DIR = "/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection/"
    DIR = 'custom_test_image_n22500_T4_p3_3'
    
    window_size = 21
    mode_covar = "ledoit-wolf"

    # Pipelines definition
    pipeline = Pipeline([
        ('rj_test', PairwiseRjTest(window_size=window_size, pvalue_rj=False, threshold=0.95, mode_covar=mode_covar))
        ],
        verbose=True)
        
    pipelines = [pipeline]
    #last characters of IN_DIR
    name = 'Pairwise change detection with sliding window and covariance matrix estimation : '+DIR
    pipelines_names = [name]

    height, width =  np.load(os.path.join(HOME_DIR+DIR, os.listdir(HOME_DIR+DIR)[0])).shape[:2]
    print(height, width)

    # Perform clustering
    results = {}
    for pipeline_name, pipeline in zip(pipelines_names, pipelines):
        print(f'Pipeline: {pipeline_name}')
        res_pipeline = pipeline.fit_predict(HOME_DIR+DIR)
        labels_pred = LabelsToImage(height, width, window_size).fit_transform(
                        res_pipeline
                )
        results[pipeline_name] = labels_pred

    # Plot data
    TRUTH_DIR = HOME_DIR+DIR+"_truth.npy"
    plot_value = np.load(TRUTH_DIR)
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(plot_value, aspect='auto')
    plt.colorbar()
    plt.title('Data')

    # Plot results
    for pipeline_name, labels_pred in results.items():
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(labels_pred, aspect='auto', cmap='gray')
        plt.title(pipeline_name)
        plt.colorbar()
        plt.show()

    # for pipeline_name, labels_pred in results.items():
    #     figure = px.imshow(labels_pred, aspect='aut>o', color_continuous_scale='gray', title="P-value for custom image 1")
    #     figure.write_html(DIR+'_palue.html', include_mathjax='cdn')
    #     figure = px.imshow(labels_pred>0.99, aspect='auto', color_continuous_scale='gray', title="(P-value > 0.99) for custom image 1")
    #     figure.write_html(DIR+'_pvalue_threshold.html', include_mathjax='cdn')

    
    # figure = px.imshow(plot_value, aspect='auto', title="True labels for custom image 1")
    # figure.write_html(DIR+'_truth.html', include_mathjax='cdn')






