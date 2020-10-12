import numpy as np
import os.path
import datetime
from sklearn.datasets import load_iris

class Configuration:

    def __init__(self):

        # Data setting 
        self.iris       = load_iris()
        self.X          = self.iris.data
        self.N          = self.X.shape[0]  # 150
        self.M          = self.X.shape[1]  # 4
        self.T          = self.iris.target

        self.use_col    = [0, 1] # [from, to]
        # 0:sepal_l, 1:sepal_w, 2:petal_l, 3:petal_w
        self.use_num    = 2
        self.Y          = self.X[:, self.use_col[0]:self.use_col[1]+1]

        # Experimental setting 
        self.max_trial  = 10
        self.max_epoch  = 100
        self.parallel   = False
        
        # K-means setting
        self.k          = 3

        # I/O setting
        self.path_out   = "./"
        now = datetime.datetime.now()
        self.log_name   = "_result_" + "K-means" +\
            "_" + str(now.year) +\
                "-" + str(now.month) +\
                    "-" + str(now.day) +\
                        "-" + str(now.hour) +\
                            "-" + str(now.minute)



    def setRandomSeed(self, seed=1):
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)



    # out config in txt
    def outSetting(self):
    
        body_setting = "+++++ Experimental Setting +++++\n"
        body_setting += "\n< Environmental Setting >\n"

        # Environment Setting
        item_env = ["trials", "epoch", "use column", "use number", "k"]
        val_env = [self.max_trial, self.max_epoch, self.use_col, self.use_num, self.k]

        for i in range(len(item_env)):
            body_setting += item_env[i].ljust(12) + ": " + str(val_env[i]) + "\n"

        path_out = self.path_out + self.log_name + "/"

        # save
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        with open( path_out +"experimental_setting.txt", "w") as f:
            f.write(body_setting)
