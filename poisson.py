from scipy import stats
import numpy as np

def poisson(lambda_h,lambda_a,points=5):
    away_prop=stats.poisson.pmf([i for i in range(points)],lambda_a)
    home_prop= stats.poisson.pmf([i for i in range(points)],lambda_h)
    k=np.outer(away_prop,home_prop)
    return k.flatten()
