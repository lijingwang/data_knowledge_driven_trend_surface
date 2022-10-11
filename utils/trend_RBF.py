# Author: Lijing Wang (lijing52@stanford.edu), 2021

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

def trend_estimation_rbf(data, x, y, smooth = 100):
    rbfi = Rbf(data['X'], data['Y'], data['data'], smooth = smooth) # you can change the smooth factor 
    xx,yy = np.meshgrid(x,y)
    data_rbf = rbfi(xx,yy)   # interpolated values
    return data_rbf 

