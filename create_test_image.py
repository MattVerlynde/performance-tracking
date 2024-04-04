##############################################################################
# Some General use functions
# Authored by Ammar Mian, 17/06/2018
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
import numpy as np
import scipy as sp
import scipy.special
import os


def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y


    
def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))


if __name__ == "__main__":
    N = 22500
    p = 3
    T = 4
    mean = np.array([1, 2, 3]) + 1j * np.array([4, 5, 6])
    covariance0 = ToeplitzMatrix(0.1, p)
    covariance1 = ToeplitzMatrix(0.9, p)
    pixels_change = 1000

    Z_tot = np.nan*np.ones((int(N**0.5), int(N**0.5), p, T)).astype(complex)
    for t in range(T):
        Z = multivariate_complex_normal_samples(mean, covariance0, N)
        Z = Z.reshape((p,int(N**0.5),int(N**0.5))).T
        Z_tot[:,:,:,t] = Z

    Z_truth = np.zeros(Z_tot.shape[:2])
    
    # for px in range(pixels_change):
    #     i,j = np.random.randint(0,int(N**0.5),2)
    #     t_change = np.random.randint(1,T)
    #     Z_truth[i,j] = t_change

    #     new_val = multivariate_complex_normal_samples(mean, covariance1, T-t_change)
    #     print(new_val.shape)

    #     Z_tot[i,j,:,t_change:] = new_val

    Zone1_change = (20,30)
    Zone2_change = (90,110)

    t1,t2 = np.random.randint(1,T),np.random.randint(1,T)

    new1 = multivariate_complex_normal_samples(mean, covariance1, (Zone1_change[0]-Zone1_change[1])**2*(T-t1)).T
    print(new1.shape)
    print(t1)
    new1 = new1.reshape((Zone1_change[1]-Zone1_change[0],Zone1_change[1]-Zone1_change[0],T-t1,3))
    print(new1.shape)
    new1 = new1.transpose((0, 1, 3, 2))
    Z_tot[Zone1_change[0]:Zone1_change[1], Zone1_change[0]:Zone1_change[1], :, t1:] = new1
    
    
    new2 = multivariate_complex_normal_samples(mean, covariance1, (Zone2_change[0]-Zone2_change[1])**2*(T-t2)).T
    new2 = new2.reshape((Zone2_change[1]-Zone2_change[0],Zone2_change[1]-Zone2_change[0],T-t2,3))
    new2 = new2.transpose((0, 1, 3, 2))
    Z_tot[Zone2_change[0]:Zone2_change[1], Zone2_change[0]:Zone2_change[1], :, t2:] = new2

    Z_truth[Zone1_change[0]:Zone1_change[1], Zone1_change[0]:Zone1_change[1]] = t1
    Z_truth[Zone2_change[0]:Zone2_change[1], Zone2_change[0]:Zone2_change[1]] = t2
    
    print(Z_tot.shape)

    # Save the image
    file = f"custom_test_image_n{N}_T{T}_p{p}_3.npy"
    np.save(file, Z_tot)
    file_truth = f"custom_test_image_n{N}_T{T}_p{p}_3_truth.npy"
    np.save(file_truth, Z_truth)

    if not os.path.exists(file[:-4]):
        os.makedirs(file[:-4])
    for t in range(T):
        np.save(f"{file[:-4]}/{file[:-4]}_t{t}.npy", Z_tot[:,:,:,t])