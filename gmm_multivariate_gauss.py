import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
np.set_printoptions(suppress = True)
import seaborn as sns

class GMM:

    def __init__(self, nclusters, dims):
        self.dims = dims
        self.nclusters = nclusters

        self.mu = np.random.randint(low = 0, high = 10, size = (self.nclusters, self.dims))
        self.pi = np.ones(self.nclusters)/self.nclusters
        self.cov = np.array([np.eye(self.dims) * 5. for _ in range(self.nclusters)])



    def e_step(self, X):
        r_ic = np.zeros((self.nclusters, X.shape[0]))
        for ii in range(self.nclusters):
            r = self.pi[ii] * multivariate_normal.pdf(X, mean = self.mu[ii], cov = self.cov[ii])
            r_ic[ii] = r
        return r_ic

    def train(self, X, epochs = 2):
        for epoch in range(epochs):
            # Expectation Step
            # (Take some randomly initialized parameters)
            r_ic = self.e_step(X)

            # Scale Margin
            # to understand relative 'influence'
            # of gauss parameters to each example
            # 
            # Normalize for all examples first 
            # to get the relative influence of each gaussian
            # for each sample
            r_ic /= r_ic.sum(axis=0)

            # Scale over Margin to understand
            # each kernels overall impact to
            # obtain 'weight' for the curren results
            mc = r_ic.sum(axis=1)

            # Maximization Step
            # (1) new kernel weights (normalize by sum)
            self.pi = mc/mc.sum()

            # (2) new covariance matrix
            for cluster in range(self.nclusters):
                # calculate the 'target' problem
                # that needs to be reduced per example
                # i.e. deviation should be minimized
                # for the most impacting cluster for
                # a given example
                #
                # This is done by calculating
                # the deviations of each cluster
                # and multiplying it by the 
                # impact weight on the 
                # respective sample
                deviation = X - self.mu[cluster] # this is our target variable (2) 'covariance'
                self.cov[cluster] = r_ic[cluster] * deviation.T @ deviation # covariance calculation * weight for sample

            # relativate/normalize the 
            # new acquainted values
            # by the overall (unnormalized!) impact, 
            # the respective clusters had 
            # on the whole dataset -> new cov 
            self.cov = self.cov / mc[:,np.newaxis,np.newaxis]

            # (3) new centers
            # just take weights per kernel per sample 
            # multiply it by their datapoint positions
            # take the impact of each datapoint and sum to get
            # new cluster centers and ...
            self.mu = (X * r_ic[:,:,np.newaxis]).sum(axis=1)
            # ...divide it by retrieved (unnormalized!) old weights, too
            # to relativize proper impact on respective prediction
            # set to new values
            self.mu = self.mu / mc[:,np.newaxis]
            if (epoch + 1) % 1 == 0:
                self.plot_gauss(X)


    def __call__(self, x):
        prediction = []
        for m, c in zip(self.mu, self.cov):
            probs = multivariate_normal.pdf(x, mean = m, cov = c)
            prediction.append(probs)
        return np.array(prediction).T


    def plot_gauss(self, X_):
        x0min, x1min = X_.min(axis=0)-10
        x0max, x1max = X_.max(axis=0)+10
        x0 = np.linspace(int(x0min), int(x0max), 50, endpoint=True)
        x1 = np.linspace(int(x1min), int(x1max), 50, endpoint=True)

        xs, ys, kind = [], [], []
        for ii in x0:
            for jj in x1:
                result = self([ii,jj])
                if result.max() > 0.00001:
                    xs.append(ii); ys.append(jj); kind.append(result.argmax().squeeze())

        plt.figure(figsize=(10,10))
        sns.kdeplot(
            x = xs,
            y = ys,
            hue = kind,
            levels = 5,
            thresh = .2,
        )
        sns.scatterplot(x = X_[:,0], y = X_[:,1])
        sns.scatterplot(x = self.mu[:,0], y = self.mu[:,1], color = 'grey', s=200, zorder = 10)
        sns.despine()
        plt.show()


if __name__ == '__main__':
    # 0. Create dataset
    np.random_state = 42
    X_,Y_ = make_blobs(cluster_std=1.5, random_state = 42, n_samples=800, centers=3)

    # Stratch dataset to get ellipsoid data
    X_ = np.dot(X_, np.random.RandomState(0).randn(2,2))

    gmm = GMM(3, 2)
    gmm.train(X_, epochs=20)
