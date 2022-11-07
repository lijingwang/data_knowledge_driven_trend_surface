# Author: Lijing Wang, 2022, lijing52@stanford.edu

import gstools as gs
import skfmm
import numpy as np
from tqdm import tqdm

def loss_function(model,data): # within class variance
    loss = np.nanvar(data[model==0])*(np.mean(model==0))+np.nanvar(data[model==1])*np.mean(model==1)
    return loss

def loss_mean_function_v2(indicators_x_alpha, phi_x_alpha, phi_x_beta, model, data): 
    #indicators_x_alpha, phi_x_alpha, 
    '''
    loss function combines logistic loss, squire mean error (bias), and mean square errors (var)
    Input
    indicators_x_alpha: borehole observed litho indicators, 1D array of 0 or 1. 
    phi_x_alpha: modelled signed distance at borehole observed locations, 1D array. 
                positive inside the target litho, negative outside the target litho
    phi_x_beta: modelled signed distance at borehole surface contact locations, 1D array. 
    Return
         loss, O_ik, O_bias, O_var
    '''
    # logistic loss function at each borehole location
    # for non-intrusive observations
    
    O_0k = (np.log(1+np.exp(phi_x_alpha[indicators_x_alpha==0]))/np.log2(2)).sum()
    # for intrusive observations
    O_1k = (np.log(1+np.exp(-phi_x_alpha[indicators_x_alpha==1]))/np.log2(2)).sum()
    O_ik = O_0k+O_1k
    
#     model_discrete  = (model>0)*1
#     O_0k = np.nanvar(data[model_discrete==0])*(np.mean(model_discrete==0))
#     O_1k = np.nanvar(data[model_discrete==1])*np.mean(model_discrete==1)
#     O_ik = O_0k+O_1k
    
    # Mean squared error at borehole contact points 
    O_bias = (np.mean(phi_x_beta))**2
    # Squared mean error at borehole 
    O_var = np.mean(phi_x_beta**2)
    #O_var = 0
    
    # Deviate from the proportion: 
    #O_prot = ()
    
    if np.isnan(O_ik):
        O_ik = 1e5
        
    loss = O_ik + O_bias + O_var
    
    #loss = O_ik*10000 + O_bias + O_var
    return loss, O_ik, O_bias, O_var
    #return loss, O_bias, O_var


def generate_m_2D(theta, x, y, seed = None):
    model = gs.Gaussian(dim=2, # 2D model
                        var= theta[1], # variance
                        len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3)], # ranges
                        angles = [theta[4]*np.pi/180] )# angle
    
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
    field = srf.structured([x, y]) + theta[0]
    return field 

def McMC_levelsets_2Dv2(model, data, 
                       loss_function=loss_mean_function_v2,  
                       sigma = 220, t_step = 0.9, iter_num = 2000, 
                       vel_range_x = [25, 35], vel_range_y = [25, 35],
                       anisotropy_ang = [0, 180],num_mp = 0):

    
    '''
    model: leveset of the intial model 
    data: data that contains the contact points
    loss_function: loss function
    sigma: hyperparameter in the likelihood function
    
    '''
    nx, ny = model.shape
    loss_array = np.zeros(iter_num)
    model_cache = np.zeros((iter_num, nx, ny))
    para_array = np.zeros((iter_num,4))

    # extract signed distance at borehole locations
    ## all the observed litho indicator
    indicators_x_alpha = data[np.isfinite(data)] 
    indicators_x_alpha[indicators_x_alpha==0.5]=1 
    ## signed dist at each borehole locations x_alpha
    phi_x_alpha_ini = model[np.isfinite(data)]
    # signed dist at contact locations x_beta
    phi_x_beta_ini = model[data==0.5]
    # calculate the loss 
    loss_prev, o_ik_prev, o_bias_prev, o_var_prev =  loss_function(indicators_x_alpha,phi_x_alpha_ini,phi_x_beta_ini, model, data)
    
    for ii in tqdm(np.arange(iter_num)):

        theta_i = np.array([0, 1, 
                            np.random.uniform(vel_range_x[0], vel_range_x[1]), 
                            np.random.uniform(vel_range_y[0], vel_range_y[1]), 
                            np.random.uniform(anisotropy_ang[0], anisotropy_ang[1])]) # you can change the range and anisotropy
        
        # create velocity fields
        velocity = generate_m_2D(theta_i, 
                                 np.arange(nx), 
                                 np.arange(ny))
        
        # levelset stochastic perturbation
        step = np.random.uniform(0,t_step)
        step_iter = np.int(np.ceil(step))
        
        for step_iter_time in range(step_iter):
            if step_iter_time<1:
                [_, F_eval] = skfmm.extension_velocities(model, velocity, dx=[1, 1],order = 1)
                dt = step/(step_iter*np.max(F_eval))
                delta_phi = dt * F_eval
                model_next = model - delta_phi # Advection
            else:
                [_, F_eval] = skfmm.extension_velocities(model_next, velocity, dx=[1, 1],order = 1)
                dt = step/(step_iter*np.max(F_eval))
                delta_phi = dt * F_eval
                model_next = model_next - delta_phi # Advection
        
        if np.sum(model_next>0)==0 or np.sum(model_next<0)==0:
            u = 1
            alpha = 0
        else:
            model_next = skfmm.distance(model_next)
            # calculate loss
            ## signed dist at each borehole locations x_alpha
            phi_x_alpha_next = model_next[np.isfinite(data)]
            # signed dist at contact locations x_beta
            phi_x_beta_next = model_next[data==0.5]    
            ## loss function
            loss_next, o_ik_next, o_bias_next, o_var_next =  loss_function(indicators_x_alpha,phi_x_alpha_next,phi_x_beta_next, model_next, data)    
        
            # acceptance ratio, alpha 
            alpha = min(1,np.exp((loss_prev**2-loss_next**2)/(sigma**2)))
            u = np.random.uniform(0, 1)

        if (u <= alpha):
            model = model_next
            loss_array[ii] = loss_next
            loss_prev = loss_next
            para_array[ii,:2] = theta_i[2:4]
            para_array[ii,2] = theta_i[4]
            para_array[ii,3] = step
        else: 
            loss_array[ii] = loss_prev
            para_array[ii,:] = -1
        
        model_cache[ii,:,:] = model
        
        if ii%20==0 and ii > 0:
            #print('Num_mp: '+str(num_mp)+', Loss function at iter '+str(ii)+': '+str(loss_prev))
            
            start = np.max([0,ii-500])
        
            print('Num_mp: '+str(num_mp)+'Accept ratio: '+str(1-np.sum(loss_array[(start+1):ii]-loss_array[start:(ii-1)]==0)/(ii-start-1))+', Loss function at iter '+str(ii)+': '+str(loss_prev))
            
        if ii%1000==0:
            np.save('results/Case2_intrusion/'+str(num_mp)+'_trend_cache.npy',model_cache[:ii,:,:])
            np.save('results/Case2_intrusion/'+str(num_mp)+'_loss_cache.npy',loss_array[:ii])
            np.save('results/Case2_intrusion/'+str(num_mp)+'_para_cache.npy',para_array[:ii,:])
        
    return [model_cache, loss_array, para_array]


def McMC_level_sets_2D(model, data, 
                       loss_function,  
                       sigma = 0.01, iter_num = 500):
    
    nx, ny = model.shape
    
    loss_array = np.zeros(iter_num)
    model_cache = np.zeros((iter_num, nx, ny))

    
    model_discrete_init = (model>0)*1
    loss_prev = loss_function(model_discrete_init,data)

    for ii in tqdm(np.arange(iter_num)):
        step = 1
        theta_i = np.array([0,1,np.random.uniform(40, 100),np.random.uniform(40, 100), np.random.uniform(0, 180)]) # you can change the range and anisotropy
        velocity = generate_m_2D(theta_i, np.arange(nx), np.arange(ny))
        
        # perturbation
        [_, F_eval] = skfmm.extension_velocities(model, velocity, dx=[1, 1],order = 1)
        
        # levelset stochastic perturbation
        dt = step/np.max(F_eval)
        delta_phi = dt * F_eval
        model_next = model - delta_phi # Advection
        if np.sum(model_next>0)==0 or np.sum(model_next<0)==0:
            break
        else:
            model_next = skfmm.distance(model_next)
        
        # loss function
        model_discrete_next = (model_next>0)*1
        loss_next = loss_function(model_discrete_next,data)
        
        # acceptance ratio, alpha 
        alpha = min(1,np.exp((loss_prev**2-loss_next**2)/(sigma**2)))
        u = np.random.uniform(0, 1)
        
        if (u <= alpha):
            model = model_next
            loss_array[ii] = loss_next
            loss_prev = loss_next
            
        else: 
            loss_array[ii] = loss_prev
        model_cache[ii,:,:] = model
        
        if ii%20==0:
            print('Loss function at iter '+str(ii)+': '+str(loss_prev))
                       
    return [model_cache, loss_array]

def mp_non_stationary_implicit_2D(args):
    [model, data, loss_function, sigma, t_step, iter_num, vel_range_x, vel_range_y,anisotropy_ang, num_mp] = args
    
    [model_cache, loss_cache, para_cache] = McMC_levelsets_2Dv2(model, data, 
                                                               loss_function,  
                                                                sigma, t_step, iter_num, 
                                                                vel_range_x, vel_range_y,
                                                                anisotropy_ang,num_mp)
    np.save('results/Case2_intrusion/'+str(num_mp)+'_model_cache.npy',model_cache)
    np.save('results/Case2_intrusion/'+str(num_mp)+'_loss_cache.npy',loss_cache)
    np.save('results/Case2_intrusion/'+str(num_mp)+'_para_cache.npy',para_cache)
    return