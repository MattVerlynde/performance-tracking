from sar_data import *
from omnibus import *
from tqdm import tqdm, trange

class Covariance(object):
    """
    Implements the covariance pooling and the Omnibus test statistic
    ENL is the (common) equivalent number of looks of the images
    """
    def __init__(self, sar_image, window_size):
        self.sar_image = sar_image
        self.ENL = window_size**2

        p = sar_image.shape[2]
        T = sar_image.shape[3]
        n = self.ENL

        self.f = (T-1)*p**2
        self.rho = 1- (2*p**2 - 1)/(6*p**(T-1)) * (T/n - 1/(n*T))

        # Omnibus test
        self.f = (T-1)*p**2
        self.rho = 1 - (2*p**2 - 1)/(6*(T-1)*p) * (T/n - 1/(n*T))
        self.w2 = p**2*(p**2-1)/(24*self.rho**2) * (T/n**2 - 1/((n*T)**2)) - (p**2*(T-1))/4 * (1 - 1/self.rho)**2

        self.lnq=np.nan*np.ones(sar_image.shape[:2])

        for i in trange(window_size//2,sar_image.shape[0]-window_size//2):
            for j in range(window_size//2, sar_image.shape[1]-window_size//2):
                Sigma_0 = np.zeros((p,p))
                result_denominator = 0
                for t in range(T):
                    Sigma_t = np.cov(sar_image[i-window_size//2:i+window_size//2+1,j-window_size//2:j+window_size//2+1,:,t].reshape(n,p).T)
                    Sigma_0 = Sigma_0 + Sigma_t
                    result_denominator = result_denominator + np.log(np.abs(np.linalg.det(Sigma_t)))
                self.lnq[i,j] = n*(p*T*np.log(T) + result_denominator - T*np.log(np.abs(np.linalg.det(Sigma_0/T))))
        
        # for i in range(0,sar_image.shape[0]-1,2):
        #     for j in range(0,sar_image.shape[1]-1,2):
        #         for image in sar_image:
        #             # self.lnq = n*(p*k*np.log(k) + sum_term - k*np.log(X.determinant()))
        #             cov_matrix = np.cov(image[i:i+2,j:j+2])

        #         X = sum(X,cov_matrix) 

        #         det = np.linalg.det(cov_matrix)
        #         sum_term = sum([np.log(Xi.determinant()) for Xi in sar_list])

        #         # self.lnq[i,j] = 

    def pvalue(self):
        "Average probability over the tested region"
        chi2 = scipy.stats.chi2.cdf
        z = -2*self.rho*self.lnq[np.logical_not(np.isnan(self.lnq))]
        return 1 - np.mean(chi2(z, df=self.f) + self.w2 * (chi2(z, df=self.f+4) - chi2(z, df=self.f)))

    def histogram(self):
        """
        Histogram of no change region
        and pdf with only chi2 term
        """

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(-2*self.lnq.flatten(), bins=100, normed=True, color="#3F5D7D")


        
if __name__ == "__main__":
    # Load data
    IN_DIR = "/home/verlyndem/Data/Selection"
    sar_image = np.load(IN_DIR+"/Scene_1.npy")

    print(sar_image.shape)

    test = Covariance(sar_image, 11)

    plt.figure()
    plt.imshow(test.lnq, aspect='auto')
    plt.colorbar() 
    plt.show()


    # print(sar_image[1:3,1:3])
    # print(sar_image[1:3,1:3].reshape(3,4*2*2))
    # print(np.cov(sar_image[1,1]).shape)

    # #reshape image to get one vector per band
    # a = np.array([[[[1,2,3,4],["a","b","c","d"],["aa","bb","cc","dd"]],[[5,6,7,8],["e","f","g","h"],["ee","ff","gg","hh"]]]])
    # print(a.shape)
    # print(a)
    # a = a.reshape(-1,3,4)
    # a = a.reshape(-1,4)
    # print(a.shape)
    # print(a)
    

    # # Covariance pooling
    # ENL = 100
    # cov = Covariance(sar_image, ENL)