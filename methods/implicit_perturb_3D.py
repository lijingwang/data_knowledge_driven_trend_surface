# Author: Lijing Wang, 2022, lijing52@stanford.edu

import gstools as gs
import skfmm
import numpy as np
from tqdm import tqdm
import sys
path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-training2/code/Users/UQ_surface"
sys.path.append(path) # might change this path

from utils.MHD import *

def generate_m_3D(theta, x, y, z, seed = None):
    model = gs.Gaussian(dim=3, 
                var= theta[1], 
                len_scale = [theta[2]/np.sqrt(3),theta[3]/np.sqrt(3),theta[4]/np.sqrt(3)],
                angles = [theta[5]*np.pi/180,theta[6]*np.pi/180])
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
    field = srf.structured([x, y, z]) + theta[0]
    return field

def loss_function_1(model,data): # within class variance
    loss = np.nanvar(data[model==0])*(np.mean(model==0))+np.nanvar(data[model==1])*np.mean(model==1)
    return loss

def loss_function_2(model,thalweg,more = False): # match the geological realism
    depth = np.argmax(model,axis = 0)
    adaptive_thresh = 50
    structure = np.array((depth<adaptive_thresh)*1)
    structure[depth==0] = 0
    if np.var(structure)==0:
        structure_skeleton = np.zeros((structure.shape[0],structure.shape[1]))
        structure_skeleton[0,0] = 1
    else:
        structure_skeleton = skeletonize(structure).astype(np.uint16)
    dis = ModHausdorffDist(np.array(np.where(structure_skeleton==1)).T,
                           np.array(np.where(thalweg==1)).T)[2]
    if more:
        return dis, depth, structure_skeleton
    else:
        return dis

    
def McMC_levelsets_3D(model, data, thalweg, loss_function_1, loss_function_2, 
                      vel_range_x = [25, 35], vel_range_y = [25, 35], vel_range_z = [25, 35], 
                      anisotropy_ang = [0, 180],
                      high_step = 1, sigma = 10, iter_num = 500, num_mp = 1):
    
    nz,ny,nx = model.shape
    
    step_array = np.zeros(iter_num)
    loss_array = np.zeros((iter_num,2))
    para_array = np.zeros((iter_num,5)) # rangex, rangey, rangez, anisotropy1, anisotropy2
    
    model_cache = np.zeros((iter_num,nz,ny,nx))
    
    model_discrete_init = (model>0)*1
    loss1_prev = loss_function_1(model_discrete_init,data)
    loss2_prev = loss_function_2(model_discrete_init,thalweg)
    loss_prev = loss1_prev + 10*loss2_prev
    
    for ii in tqdm(np.arange(iter_num)):
        step_i  = uniform(low = 0, high = high_step, size = 1)[0] # you can change the step size
        
        theta_i = np.array([0, 1, 
                            np.random.uniform(vel_range_x[0], vel_range_x[1]), 
                            np.random.uniform(vel_range_y[0], vel_range_y[1]), 
                            np.random.uniform(vel_range_z[0], vel_range_z[1]), 
                            np.random.uniform(anisotropy_ang[0], anisotropy_ang[1]),
                            np.random.uniform(anisotropy_ang[0], anisotropy_ang[1])]) # you can change the range and anisotropy
        
        # create velocity fields
        velocity = generate_m_3D(theta_i, 
                                 np.arange(nx), 
                                 np.arange(ny),
                                 np.arange(nz))
        
        velocity = velocity.T
        
        # perturbation
        [_, F_eval] = skfmm.extension_velocities(model, velocity,dx=[1, 1, 1],order = 1)
        
        dt = step_i/np.max(F_eval)
        delta_phi = dt * F_eval
        model_next = model - delta_phi #Advection
        model_next = skfmm.distance(model_next)
        
        # loss function
        model_discrete_next = (model_next>0)*1
        loss1_next = loss_function_1(model_discrete_next,data)
        loss2_next = loss_function_2(model_discrete_next,thalweg)
        
        loss_next = loss1_next + 10*loss2_next
        
        # acceptance ratio, alpha 
        alpha = min(1,np.exp((loss_prev**2-loss_next**2)/(sigma**2)))
        
        u = np.random.uniform(0, 1)
        
        if (u <= alpha):
            model = model_next
            loss_array[ii,:] = [loss1_next,loss2_next]
            step_array[ii]     = step_i
            para_array[ii,:]   = theta_i[2:]
            loss_prev = loss_next
            loss1_prev = loss1_next
            loss2_prev = loss2_next
        else: 
            loss_array[ii,:] = [loss1_prev,loss2_prev]
            step_array[ii]   = -1
            para_array[ii]   = -1

        model_cache[ii,:,:,:] = model
        
        if ii%20==0 and ii > 0:
            start = np.max([0,ii-500])
            print('Num_mp: '+str(num_mp)+'Accept ratio: '+str(1-np.sum(loss_array[(start+1):ii,0]-
                                                                       loss_array[start:(ii-1),0]==0)/(ii-start-1))+', Loss function at iter '+str(ii)+': '+str(loss_prev))
            
        if ii%20==0:
            np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_trend_cache.npy',model_cache[:ii,:,:])
            np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_loss_cache.npy',loss_array[:ii,:])
            np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_step_cache.npy',step_array[:ii])
            np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_para_cache.npy',para_array[:ii,:])

    return [model_cache, loss_array, step_array, para_array]


def mp_non_stationary_implicit_3D(args):
    [model, data, thalweg, loss_function_1, loss_function_2, vel_range_x, vel_range_y,vel_range_z,anisotropy_ang, high_step, sigma, iter_num, num_mp] = args
    
    [model_cache, loss_cache, step_cache, para_cache] = McMC_levelsets_3D(model, data, thalweg, loss_function_1, loss_function_2, 
                                                                          vel_range_x, vel_range_y, vel_range_z, 
                                                                          anisotropy_ang,
                                                                          high_step, sigma, iter_num, num_mp)
    np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_model_cache.npy',model_cache)
    np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_loss_cache.npy',loss_cache)
    np.save('../results/Case3_palaeovalley/'+str(num_mp)+'_para_cache.npy',para_cache)
    return
