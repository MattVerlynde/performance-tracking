# Useful functions to load and process data

from collections.abc import Generator
import os
import glob
from typing import Tuple, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from math import ceil
from sklearn.base import BaseEstimator, TransformerMixin

from numpy.typing import ArrayLike
from joblib import Parallel, delayed
import tqdm

import scipy.stats

# Dataset helpers
# -----------------------------------------------------------------------------
class SARImageTimeSeriesReader:
    """Helper class for reading SAR Image Time series split into multiple
    files with format .npy and same sahpe at each time.

    Attributes
    ----------
    path: str
        Folder in which all files are stored
    preload: bool, default=false
        Whether to preload all the series if enough RAM is available
    verbose: int, default=0
        Verbosity level
    crop_indexes, list
        list of 4 values: [beg_row, end_row, beg_col, end_row]
    """

    def __init__(self,
                 path: str,
                 preload: bool = False,
                 verbose: int = 0,
                 crop_indexes: List = None) -> None:
        self.path = path
        self.preload = preload
        self.verbose = verbose
        self.data = None
        self.crop_indexes = crop_indexes

        # Finding list of all .npy files in path
        self.list_image_paths = glob.glob(os.path.join(self.path, "*.npy"))
        self.list_image_paths.sort()
    
        if self.preload:
            print(f"Preloading SITS from {self.path}")
            self._preload_series()

    def _preload_series(self) -> None:
        """Preaload all the series into memory"""

        if self.verbose > 0:
            pbar = tqdm.tqdm(total=len(self.list_image_paths))

        self.data = []
        for file_path in self.list_image_paths:

            temp = np.load(file_path)
            if self.crop_indexes is not None:
                beg_row, end_row, beg_col, end_col = self.crop_indexes
                temp = temp[beg_row:end_row, beg_col:end_col]
            self.data.append(temp)

            if self.verbose > 0:
                pbar.update()
                pbar.set_description(f'Read {file_path}', refresh=True)

        self.data = np.stack(self.data, axis=0)

        if self.verbose > 0:
            pbar.close()

    def __len__(self) -> int:
        return len(self.list_image_paths)

    def __str__(self) -> str:
        return f"SARImageTimeSeriesReader(path={self.path}, "+\
            f"preload={self.preload}, verbose={self.verbose}, crop={self.crop_indexes})"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key) -> ArrayLike:
        """Get a slice of SITS over time."""

        # We can just slice on the array if preloaded
        if self.preload:
            return self.data[key]
    
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            assert key < len(self) and key >= 0, \
                f"Index {key} is out of range for images in the stack {len(self)}"
            if self.verbose > 1:
                print(f"Reading {self.list_image_paths[key]}")
            temp = np.load(self.list_image_paths[key])
            if self.crop_indexes is not None:
                beg_row, end_row, beg_col, end_col = self.crop_indexes
                temp = temp[beg_row:end_row, beg_col:end_col]
            return temp

        elif isinstance(key, slice):
            assert key.start >= 0  and key.start <= len(self),\
                f"Slice start {key.start} is out of bounds!"
            assert key.stop >= 0 and key.stop <= len(self),\
                f"Slice stop {key.stop} is out of bounds!"
            if key.step is None:
                step = 1
            else:
                step = key.step
            if self.verbose > 1:
                print(f"Reading files {self.list_image_paths[key]}")
                iterator = tqdm.trange(key.start, key.stop, step)
            else:
                iterator = range(key.start, key.stop, step)
            if self.crop_indexes is not None:
                beg_row, end_row, beg_col, end_col = self.crop_indexes
                data = [
                    np.load(self.list_image_paths[ii])[beg_row:end_row, beg_col:end_col]
                    for ii in iterator]
            else:
                data = [np.load(self.list_image_paths[ii]) for ii in iterator]
            return np.stack(data, axis=0)

        else:
            raise TypeError("Invalid argument type.")

    def iter_twobytwo(self) -> Generator:
        """Memory efficient iterable that yiels image by pair of
        two from start to end"""

        def generate():
            cached_image = self[0]
            for t in range(1, len(self)):
                new_image = self[t]
                yield (cached_image, new_image)
                cached_image = new_image

        return generate()



# Processing helpers
# -----------------------------------------------------------------------------
class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """Sliding window for three-dimensional data.

    Parameters
    ----------
    window_size : int
        Size of the sliding window.
    overlap : int, default=0
        Overlap between windows.
    """

    def __init__(self, window_size: int, overlap: int = 0) -> None:
        assert window_size % 2 == 1, "Window size must be odd."
        assert overlap >= 0, "Overlap must be positive."
        assert overlap <= window_size//2, \
            "Overlap must be smaller or equal than int(window_size/2)."
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ArrayLike, y=None) -> None:
        """Keep in memory n_rows and n_columns of original data.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input image

        Returns
        -------
        self : SlidingWindowVectorize instance
            The SlidingWindowVectorize instance.
        """
        self.n_rows, self.n_columns, _ = X.shape
        return self

    def __len__(self) -> int:
        return (self.n_rows-self.window_size+1) * \
            (self.n_columns-self.window_size+1) // self.overlap^2

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform original data with a sliding window.

        Transform original three-dimensional data into a sliding window view
        over the first two dimensions.

        Parameters
        ----------
        X : ndarray, shape (n_rows, n_columns, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray, shape (n_pixels, window_size**2, n_features)
            Output data, with n_pixels = (n_rows-window_size+1) x
            (n_columns-window_size+1) // overlap^2
        """
        X = sliding_window_view(
            X,
            window_shape=(self.window_size, self.window_size),
            axis=(0, 1),
        )
        if self.overlap is not None:
            if self.overlap > 0:
                X = X[::self.overlap, ::self.overlap]
        else:
            X = X[::self.window_size//2, ::self.window_size//2]
            self.overlap = self.window_size//2

        # reshape to (n_pixels, n_samples, n_features) with
        # n_pixels = axis0*axis1, n_samples = axis3*axis_4, n_features = axis2
        # print(f"X.shape: {X.shape}")
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3]*X.shape[4]))
        X = X.transpose((0, 2, 1))
        return X

    def fit_transform(self, X: ArrayLike, y=None) -> ArrayLike:
        return self.fit(X).transform(X)

    def inverse_predict(self, y: ArrayLike) -> ArrayLike:
        """Transform predictions over sliding windows back to original data.

        Transform the predictions over sliding windows data back to original
        data shape.

        Parameters
        ----------
        y : ndarray, shape (n_pixels,)
            Predictions.

        Returns
        -------
        X : ndarray, shape (n_new_rows, n_new_columns)
            Output predicted data, with n_new_rows = (n_rows-window_size+1) //
            overlap and n_new_columns = (n_columns-window_size+1) // overlap.
        """
        # compute reshape size thanks to window_size before overlap
        n_new_rows = self.n_rows - self.window_size + 1
        n_new_columns = self.n_columns - self.window_size + 1

        # take into account overlap
        if self.overlap > 0:
            n_new_rows = ceil(n_new_rows/self.overlap)
            n_new_columns = ceil(n_new_columns/self.overlap)

        return y.reshape((n_new_rows, n_new_columns))


def gaussian_rj_statistic(
    j, data_j, X_1tojminus1, k) -> Tuple[float, ArrayLike, ArrayLike]:
    """Gaussian test statistic for change-detection. Online version.

    Parameters
    ----------
    j: int
        Date at which we compute the test statistic.
    data_j: array of shape (n_samples, n_features)
        Unnormalized covariance at time j
    X_jminus1: array of shape (n_features, n_features)
        Unnormalized covariance at time j-1.
    X_1tojminus1: array of shape (n_features, n_features)
        Accumulated covariance over time

    Returns
    -------
    lnRj: float
        Current value of test statistic.
    X_1toj: array of shape (n_features, n_features)
        New accumulated covariance.
    """
    n, p = data_j.shape
    X_j = np.cov(data_j.T, rowvar=True, bias=False)*n*k
    X_1toj = X_1tojminus1 + X_j
    lnRj = n*(p*(j*np.log(j) - (j-1)*np.log(j-1)) +\
        (j-1)*np.log(np.abs(np.linalg.det(X_1tojminus1))) +\
        np.log(np.abs(np.linalg.det(X_j))) -\
        j*np.log(np.abs(np.linalg.det(X_1toj))))
    return np.real(lnRj), X_1toj


class GaussianChangeDetector(BaseEstimator, TransformerMixin):
    """Pairwise Rj test for change detection using the covariance matrix of the
    SAR image between two dates."""

    def __init__(self,
                 window_size: int,
                 n_jobs: int = 1,
                 verbose: int = 0):
        self.window_size=window_size
        self.ENL = window_size**2
        self.n_jobs = n_jobs
        self.lnQ = None
        self.verbose = verbose


    def fit(self, X: SARImageTimeSeriesReader, y=None) -> None:
        """Estimate change detection probabilities of image in input."""

        sliding_window = SlidingWindowVectorize(self.window_size)

        # Loop over time two by two
        time_iterator = X.iter_twobytwo()
        lnQ_all = 0 
        self.parameters_predict = (len(X), self.ENL, X[0].shape[2])
        for j in range(2, len(X)+1):

            I_jminus1, I_j = next(time_iterator)

            # Sliding window over pixels
            Sw_j = sliding_window.fit_transform(I_j)

            # Initialize accumulated unormalized covariances at beginning
            # I do it here cause I don"t know the size of the data beforehand
            if j == 2:

                Sw_jminus1 = sliding_window.fit_transform(I_jminus1)
                # print(Sw_jminus1[0])
                # print(f"Sw_jminus1[0].shape {Sw_jminus1[0].shape}")
                # print(Sw_jminus1[0])
                # print(f"Sw_jminus1[0].T.shape {Sw_jminus1[0].T.shape}")
                # print(Sw_jminus1[0].T)

                if self.verbose:
                    print("Computing covariances at date j=1")
                    iterate_pixels = tqdm.tqdm(Sw_jminus1)
                    # print(f"Sw_jminus1.shape {Sw_jminus1.shape}")
                else:
                    iterate_pixels = Sw_jminus1
                X_1_all = Parallel(n_jobs=self.n_jobs)(
                        delayed(np.cov)(x.T, rowvar=True, bias=False)
                        for x in iterate_pixels
                )

                X_1tojminus1_all = np.stack(X_1_all, axis=0)*Sw_j.shape[1]*len(X)
                print(f"X_1tojminus1_all.shape {X_1tojminus1_all.shape}")
            
            # Process all pixels
            iterate_pixels = zip(Sw_j, X_1tojminus1_all)
            if self.verbose > 0:
                print(f"Adding cd betwenn dates {j-1} and {j}")
                iterate_pixels = tqdm.tqdm(iterate_pixels, total=len(Sw_j))
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(gaussian_rj_statistic)(j, data_j, X_1tojminus1, len(X))
                for data_j, X_1tojminus1 in iterate_pixels
            )
            temp = np.array([x[0] for x in results])
            X_1tojminus1_all = np.stack([x[1] for x in results])
            lnQ_all += sliding_window.inverse_predict(np.array(temp))
        self.lnQ_all = lnQ_all

        return self

    def predict(self, X: ArrayLike):
        chi2 = scipy.stats.chi2.cdf
        T, n, p = self.parameters_predict
        print(f"Parameters predict {T} {n} {p}")
        
        f = (T-1)*p**2
        rho = 1 - (2*p**2-1)/(6*(T-1)*p)*(T/n-1/(n*T))
        omega_2 = (p**2)*(p**2-1)/(24*rho**2)*(T/(n**2)-(1/((n*T)**2))) -\
                (p**2)*(T-1)/4 * (1 - 1/rho)**2
        Z = -2*rho*self.lnQ_all
        print(f"rho {rho}, omega_2 {omega_2}, f {f}, Z {Z}")
        
        return (chi2(Z, df=f) + omega_2*(chi2(Z, df=f+4) - chi2(Z, df=f)))
    
    def fit_predict(self, X: SARImageTimeSeriesReader, y=None):
        return self.fit(X).predict(X)