import numpy            as np
import configuration    as cf
import k_means          as km
import logger           as lg
from joblib             import Parallel, delayed

def run(opt, cnf, i):
    if cnf.parallel == True:
        cnf.setRandomSeed(seed=i+1)
    opt.initialization()
    opt.do()
    opt.out_graph()

if __name__ == '__main__':
    cnf = cf.Configuration()
    cnf.outSetting()
    log = lg.Logger(cnf)
    log.out_graph()
    if cnf.parallel == True:
        opt = km.KMEANS(cnf, log) 
        Parallel(n_jobs=-1)([delayed(run)(opt, cnf, i) for i in range(cnf.max_trial)])
    else:
        for i in range(cnf.max_trial):
            cnf.setRandomSeed(seed=i+1)
            opt = km.KMEANS(cnf, log) 
            run(opt, cnf, 0)
            del opt
    log.out_epoch()
