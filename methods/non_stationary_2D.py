# load necessary modules
import numpy as np
from tqdm import tqdm
from numpy.random import uniform as uniform
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from scipy import stats
import gstools as gs

def generate_m_2D(theta, x, y, seed = None):
    model = gs.Gaussian(dim=2, 
                var= theta[1], 
                len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3)],
                angles = theta[4]*np.pi/180)
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
        
    field = srf.structured([x, y]) + theta[0]
    return field

def residual_quantile(trend, data, data_idx, outlier_proportion = 0.05):
    residual = (data-trend)[data_idx]
    ecdf = ECDF(residual)
    norm_quantile = norm.ppf(ecdf(residual))
    norm_quantile[norm_quantile==np.inf] = 5 # a large value for N(0,1)

    res = stats.linregress(norm_quantile, residual)
    deviation = np.abs(res.intercept + res.slope*norm_quantile-residual)
    threshold = np.quantile(deviation,1-outlier_proportion)
    
    loss = np.mean(np.square(residual[deviation<threshold]))
    
    outlier_matrix = np.zeros(data.shape)
    outlier_matrix[data_idx] = (deviation>=threshold)*1
    
    return norm_quantile,residual,deviation,threshold,outlier_matrix


def loss_function(trend, data, data_idx, var_ratio_trend = 0.5, outlier_proportion = 0.05):
    # Calculate the residual given the current trend
    residual = (data-trend)[data_idx]
    
    # See if we have any outlier
    ecdf = ECDF(residual)
    norm_quantile = norm.ppf(ecdf(residual))
    norm_quantile[norm_quantile==np.inf] = 5 # a large value for N(0,1)
    res = stats.linregress(norm_quantile, residual)
    
    deviation = np.abs(res.intercept + res.slope*norm_quantile-residual)
    threshold = np.quantile(deviation,1-outlier_proportion)
    loss1 = np.mean(np.square(residual[deviation<threshold]))
    loss2 = np.square(np.mean(residual[deviation<threshold]))
    #loss3 = np.abs(np.var(data[data_idx][deviation<threshold])*var_ratio_trend-np.var(trend))
    
    loss = loss1+loss2 #+loss3
    return loss

def loss_function_hard_data(trend, data, data_idx):
    # Calculate the residual given the current trend
    residual = (data-trend)[data_idx]
    loss1 = np.mean(np.square(residual))
    loss2 = np.square(np.mean(residual))
     
    loss = loss1+loss2 
    return loss

def non_stationary_2D(trend, data, loss_prev, x, y, range_min = [50,50], range_max = [100,100],
                           high_step = 1, sigma = 0.05, iter_num = 500):
    data_idx = ~np.isnan(data)
    loss_cache  = np.zeros((iter_num,1))
    step_cache  = np.zeros((iter_num,1))
    
    trend_cache = np.zeros((iter_num, trend.shape[0], trend.shape[1]))
    
    loss_prev = loss_function_hard_data(trend, data, data_idx)
    
    for i in tqdm(range(iter_num)):
        step_i  = uniform(low = 1, high = high_step, size = 1)[0] # you can change the step size
                
        rangeMax_i = uniform(low = range_min[0], high = range_max[0], size = 1)[0]
        rangeMin_i = uniform(low = range_min[1], high = range_max[1], size = 1)[0]
        angle_i = uniform(low = 0, high = 180, size = 1)[0]
        
        velo_i = generate_m_2D([0,1,rangeMax_i,rangeMin_i,angle_i], x, y, seed = None)

        # perturbation
        trend_next = trend + step_i*velo_i

        # loss function
        loss_next = loss_function_hard_data(trend_next, data, data_idx)

        # sigma (hyperparameter)
        sigma = sigma
        
        # acceptance ratio, alpha 
        alpha = min(1,np.exp((loss_prev**2-loss_next**2)/(2*sigma**2)))
        
        # accept or not, save cache
        u = uniform(size = 1)
        if (u <= alpha):
            trend = trend_next
            loss_cache[i,:] = loss_next
            step_cache[i]     = step_i
            loss_prev = loss_next
        else: 
            loss_cache[i,:] = loss_prev
            step_cache[i]     = np.nan
        
        trend_cache[i,:,:] = trend
        
        if i%100 == 0:
            print('iter'+str(i)+': '+str(loss_cache[i,:]))
    return [trend_cache, loss_cache]

def mp_non_stationary_2D(args):
    [trend, data, loss_prev, x,y, range_min, range_max, high_step, sigma,  iter_num, num_mp] = args
    [trend_cache, loss_cache] = non_stationary_2D(trend, data, loss_prev, x, y, range_min, range_max,
                                                  high_step = high_step, sigma = sigma, iter_num = iter_num)
    np.save('../results/McMC_2D/'+str(num_mp)+'_trend_cache.npy',trend_cache)
    np.save('../results/McMC_2D/'+str(num_mp)+'_loss_cache.npy',loss_cache)
    return