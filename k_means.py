from copy import deepcopy
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt

class KMEANS:

    def __init__(self, cnf, log):
        self.cnf        = cnf
        self.log        = log
        self.path_out   = self.cnf.path_out
        self.path_out   += '/{0}/'.format(self.cnf.log_name)
        self.path_trial = self.path_out + 'trials' 
        self.w          = np.zeros((self.cnf.k, self.cnf.use_num))
        self.new_w      = np.zeros((self.cnf.k, self.cnf.use_num))
        self.clust      = np.zeros(self.cnf.N)


    def initialization(self):
        self.w      = self.cnf.Y[self.cnf.rd.choice(self.cnf.N, self.cnf.k)]

    def do(self):
        for epo in range(self.cnf.max_epoch):
            for i in range(self.cnf.N):
                d = []
                for j in range(self.cnf.k):
                    d.append(np.linalg.norm(self.w[j] - self.cnf.Y[i]))
                self.clust[i] = d.index(min(d))
            for j in range(self.cnf.k):
                self.new_w[j] = self.cnf.Y[self.clust==j].mean(axis=0)
            print(np.sum(self.new_w == self.w))
            if np.sum(self.new_w == self.w) == self.cnf.k * self.cnf.use_num:
                print("converged")
                self.log.count_epoch(epo)
                break
            else:
                self.w = deepcopy(self.new_w)
                if epo == self.cnf.max_epoch - 1:
                    self.log.count_epoch(epo)

    def out_graph(self):

        fig = plt.figure()

        ax = fig.add_subplot(1,1,1)

        ax.scatter(np.ravel(self.cnf.Y[:, 0:1]), np.ravel(self.cnf.Y[:, 1:2]), c=np.ravel(self.clust), cmap="autumn", label='data')

        ax.scatter(np.ravel(self.w[:, 0:1]), np.ravel(self.w[:, 1:2]), marker='x', label='weighted center')

        ax.set_title('k-means class --- trial {}'.format(self.cnf.seed))
        ax.set_xlabel(self.cnf.iris.feature_names[self.cnf.use_col[0]:self.cnf.use_col[0]+1][0])
        ax.set_ylabel(self.cnf.iris.feature_names[self.cnf.use_col[1]:self.cnf.use_col[1]+1][0])
        ax.legend()

        fig.savefig(self.path_trial + '/trial{}.png'.format(self.cnf.seed) , dpi=150)
