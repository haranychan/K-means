
import os
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt


class Logger:
    def __init__(self, cnf):
        self.dat, self.cnf = [], cnf
        self.path_out = cnf.path_out
        self.path_out += '/{0}/'.format(self.cnf.log_name)
        self.path_trial = self.path_out + 'trials' 
        if not os.path.isdir(self.path_trial):
            os.makedirs(self.path_trial)


    def out_graph(self):

        fig = plt.figure()

        ax = fig.add_subplot(1,1,1)

        ax.scatter(np.ravel(self.cnf.Y[:, 0:1]), np.ravel(self.cnf.Y[:, 1:2]), c=np.ravel(self.cnf.T), cmap="autumn")
        
        ax.set_title('true class')
        ax.set_xlabel(self.cnf.iris.feature_names[self.cnf.use_col[0]:self.cnf.use_col[0]+1][0])
        ax.set_ylabel(self.cnf.iris.feature_names[self.cnf.use_col[1]:self.cnf.use_col[1]+1][0])

        fig.savefig(self.path_out + "true_class.png" , dpi=150)

    def count_epoch(self, epo):
        self.dat.append(epo+1)

    def out_epoch(self):
        self.dat = [self.dat]
        head = ','.join(["trial{}".format(i+1) for i in range(self.cnf.max_trial)])
        np.savetxt(self.path_out + '/epoch.csv', self.dat, delimiter=',', header = head, comments = '')      
        self.dat = []

