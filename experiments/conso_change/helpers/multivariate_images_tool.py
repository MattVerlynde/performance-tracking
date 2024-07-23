##############################################################################
# Some functions useful for treatments on multivariate images
# Authored by Ammar Mian, 09/11/2018
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
from multiprocessing import Process, Queue
import numpy as np
import time
from tqdm import tqdm
import warnings

def tyler_estimator_covariance(𝐗, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p, N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)

def tyler_estimator_covariance_matandtext(𝐗, tol=0.0001, iter_max=20):
    """ A function that computes the Modified Tyler Fixed Point Estimator for 
    covariance matrix estimation under problem MatAndText.
        Inputs:
            * 𝐗 = a matrix of size p*N*T with each saptial observation along column dimension and time
                observation along third dimension.
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    (p, N, T) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and iteration < iter_max:

        # Compute the textures for each pixel using all the dates avalaibe
        τ = 0
        i𝚺 = np.linalg.inv(𝚺)
        for t in range(0, T):
            τ = τ + np.diagonal(𝐗[:,:,t].conj().T@i𝚺@𝐗[:,:,t])

        # Computing expression of the estimator
        𝚺_new = 0
        for t in range(0, T):
            𝐗_bis = 𝐗[:,:,t] / np.sqrt(τ)
            𝚺_new = 𝚺_new + (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')

        # Updating 𝚺
        𝚺 = 𝚺_new
        iteration = iteration + 1

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)

def scale_and_shape_equality_robust_statistic(𝐗, args):
    """ GLRT test for testing a change in the scale or/and shape of 
        a deterministic SIRV model.
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = tol, iter_max for Tyler, scale
        Outputs:
            * the statistic given the observations in input"""

    (p, N, T) = 𝐗.shape
    (tol, iter_max, scale) = args

    # Estimating 𝚺_0 using all the observations
    (𝚺_0, δ, niter) = tyler_estimator_covariance_matandtext(𝐗, tol, iter_max)
    i𝚺_0 = np.linalg.inv(𝚺_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(𝚺_0)))
    log_denominator_determinant_terms = 0
    𝛕_0 = 0
    log𝛕_t = 0
    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating 𝚺_t
        (𝚺_t, δ, iteration) = tyler_estimator_covariance(𝐗[:,:,t], tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing texture estimation
        𝛕_0 =  𝛕_0 + np.diagonal(𝐗[:,:,t].conj().T@i𝚺_0@𝐗[:,:,t]) / T
        log𝛕_t = log𝛕_t + np.log(np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = T*p*np.sum(np.log(𝛕_0))
    log_denominator_quadtratic_terms = p*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ 


def sliding_windows_treatment_image_time_series(image, windows_mask, function_to_compute, function_args, multi=False, queue=0,
                                                progressbar=False):
    """ A function that allowing to compute a sliding windows treatment over a multivariate
        image time series.
        Inputs:
            * image = a numpy array of shape (n_r,n_c,p,T) where n_r is the number of rows,
                      n_c is the number of columns, p is the number of canals and T is the length
                      of the time series.
            * windows_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * function_args = arguments to pass to function_to_compute
            * multi = True if parallel computing (use the parallel function not this one), False if not
            * queue = to obtain result for parallel computation
            * progressbar = display a progressbar or not
        Outputs:
            * a 3-d array corresponding to the results. First two dimensions are spatial while the third correspond
              to the output of function_to_compute."""

    n_r, n_c, p, T = image.shape
    m_r, m_c = windows_mask.shape
    N = m_r*m_c
    result = []
    if not progressbar:
        for i_r in range(int(m_r/2),n_r-int(m_r/2)): # Iterate on rows
            result_line = []
            for i_c in range(int(m_c/2),n_c-int(m_c/2)): # Iterate on columns
    
                # Obtaining data corresponding to the neighborhood defined by the mask
                local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, 0].T.reshape((p,N))
                for t in range(1,T):
                    local_data = np.dstack((local_data, image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, t].T.reshape((p,N))))
                
                # Applying mask
                local_data = local_data.reshape((p,N,T))
                local_data = local_data[:,windows_mask.reshape(m_r*m_c).astype(bool),:]
    
                # Computing the function over the local data
                result_line.append(function_to_compute(local_data, function_args))
            result.append(result_line)
    else:
        for i_r in tqdm(range(int(m_r/2),n_r-int(m_r/2))): # Iterate on rows
            result_line = []
            for i_c in range(int(m_c/2),n_c-int(m_c/2)): # Iterate on columns
    
                # Obtaining data corresponding to the neighborhood defined by the mask
                local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, 0].T.reshape((p,N))
                for t in range(1,T):
                    local_data = np.dstack((local_data, image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, t].T.reshape((p,N))))
                
                # Applying mask
                local_data = local_data.reshape((p,N,T))
                local_data = local_data[:,windows_mask.reshape(m_r*m_c).astype(bool),:]
    
                # Computing the function over the local data
                result_line.append(function_to_compute(local_data, function_args))
            result.append(result_line)
        

    if multi:
        queue.put(result)
    else:
        return np.array(result)


def sliding_windows_treatment_image_time_series_parallel(image, windows_mask, function_to_compute, function_args,
                                                multi=False, number_of_threads_rows=3, number_of_threads_columns=3,
                                                progressbar=False):
    """ A function that is a prallelisation of sliding_windows_treatment_image_time_series
        Inputs:
            * image = a numpy array of shape (n_r,n_c,p,T) where n_r is the number of rows,
                      n_c is the number of columns, p is the number of canals and T is the length
                      of the time series.
            * windows_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * function_args = arguments to pass to function_to_compute
            * multi = True if parallel computing, False if not
            * number_of_threads_columns = number of thread to use in columns 
                (total threads = number of cores of the machine in general)
            * number_of_threads_rows = number of thread to use in columns 
                (total threads = number of cores of the machine in general) 
            * progressbar = display a progressbar or not
        Outputs:
            * number_of_threads_columns = number of thread to use in columns 
                (total threads = number of cores of the machine in general)"""

    if multi:
        # Slicing original image while taking into accound borders effects
        n_r, n_c, p, T = image.shape
        m_r, m_c = windows_mask.shape
        image_slices_list = [] # Will contain each slice
        for i_row in range(0,number_of_threads_rows):
            # Indexes for the sub_image for rows
            if i_row == 0:
                index_row_start = 0
            else:
                index_row_start = int(n_r/number_of_threads_rows)*i_row - int(m_r/2)
            if i_row == number_of_threads_rows-1:
                index_row_end = n_r
            else:
                index_row_end = int(n_r/number_of_threads_rows)*(i_row+1) + int(m_r/2)

            # Slices for each row
            image_slices_list_row = []
            for i_column in range(0, number_of_threads_columns):
                # Indexes for the sub_image for colums
                if i_column == 0:
                    index_column_start = 0
                else:
                    index_column_start = int(n_c/number_of_threads_columns)*i_column - int(m_c/2)
                if i_column == number_of_threads_columns-1:
                    index_column_end = n_c
                else:
                    index_column_end = int(n_c/number_of_threads_columns)*(i_column+1) + int(m_c/2)

                # Obtaining each slice and putting it in the list
                image_slice = image[index_row_start:index_row_end, index_column_start:index_column_end, :, :]
                image_slices_list_row.append(image_slice)

            # 2d list of slices
            image_slices_list.append(image_slices_list_row)

        # Freeing space
        image_slice = None
        image_slices_list_row = None

        # Serves to obtain result for each thread
        queues = [[Queue() for i_c in range(number_of_threads_columns)] for i_r in range(number_of_threads_rows)]

        # Arguments to pass to each thread
        args = [(image_slices_list[i_r][i_c], windows_mask, function_to_compute, function_args, 
                True, queues[i_r][i_c], progressbar) for i_r in range(number_of_threads_rows) for i_c in range(number_of_threads_columns)] 

        # Initialising the threads
        jobs = [Process(target=sliding_windows_treatment_image_time_series, args=a) for a in args]

        # Starting parallel computation
        for j in jobs: j.start()

        # Obtaining result for each thread
        results_list = [] # Results container
        for i_r in range(0,number_of_threads_rows):
            results_row_list = []
            for i_c in range(0,number_of_threads_columns):
                results_row_list.append( queues[i_r][i_c].get() )
            results_list.append(results_row_list)
        results_row_list = None

        # Waiting for each thread to terminate
        for j in tqdm(jobs): j.join()

        # Now we reform the resulting image from the slices of results
        results = []
        for i_r in range(0,number_of_threads_rows):
            final_array_row = []
            for i_c in range(0,number_of_threads_columns):
                final_array_row.append(results_list[i_r][i_c])
            results.append(np.hstack(final_array_row))
        results = np.vstack(results)
        final_array_row = None

    else:
        results = sliding_windows_treatment_image_time_series(image, windows_mask = windows_mask, 
                                                              function_to_compute = function_to_compute, 
                                                              function_args = function_args, 
                                                              multi = multi,
                                                              progressbar = progressbar)
    return results
   
    