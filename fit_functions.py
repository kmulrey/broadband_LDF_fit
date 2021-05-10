import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
import glob
from scipy.interpolate import interp1d
import re
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, report_fit
from scipy import special
import random
import scipy.interpolate as interpolate
import sim_helper as helper



def objective_gm(params, r, data,Egeo_prime,p):
    resid_0 = 0.0*data[:]
    error=0.1*np.max(data)
    sigma=error*np.ones([len(data)])
    resid_0 = (data - return_gm_fit(params,r,Egeo_prime,p))**2/sigma**2

    return resid_0.flatten()
    
def returnNminus(R,sigma):
    return sigma*np.pi*np.sqrt(2)*(np.sqrt(np.pi)*R*special.erfc(-1*R/(np.sqrt(2)*sigma))+np.sqrt(2)*sigma*np.exp(-1*R**2/(2*sigma**2)))
    
def returnNplus(R,sigma):
    return 2*np.pi*sigma*(np.sqrt(2*np.pi)*R*special.erf(R/(np.sqrt(2)*sigma))+2*sigma*np.exp(-1*R**2/(2*sigma**2))                     )
    
def returnNce(k,sigma):
    return (2*np.pi/(k+1))*np.power(2,k)*np.power(2*k+2,-0.5*k)*np.power(sigma,k+2)*special.gamma(k/2+1)
    
def return_gm_fit(params,r,Egeo_prime,p):
    r=np.abs(r)

    Rgeo=params['Rgeo'].value
    sigma=params['sigma'].value
    #print(Rgeo,sigma)
    if Rgeo>=0:
        N=returnNplus(Rgeo,sigma)
    else:
        N=returnNminus(Rgeo,sigma)
    fgeo=0
    if Rgeo<0:
        fgeo=(1/N)*(Egeo_prime)*np.exp(-1.0*np.power(((r-Rgeo)/(np.sqrt(2)*sigma)),p))
    if Rgeo>=0:
        fgeo=(1/N)*(Egeo_prime)*(np.exp(-1.0*np.power(((r-Rgeo)/(np.sqrt(2)*sigma)),p))+np.exp(-1.0*np.power(((r+Rgeo)/(np.sqrt(2)*sigma)),p)))
    return fgeo
    
def return_gm(r,Egeo_prime,Rgeo,sigma,p):
    r=np.abs(r)
    fgeo=0
    if Rgeo>=0:
        N=returnNplus(Rgeo,sigma)
    else:
        N=returnNminus(Rgeo,sigma)

    if Rgeo<0:
        fgeo=(1/N)*(Egeo_prime)*np.exp(-1.0*np.power(((r-Rgeo)/(np.sqrt(2)*sigma)),p))
    if Rgeo>=0:
        fgeo=(1/N)*(Egeo_prime)*(np.exp(-1.0*np.power(((r-Rgeo)/(np.sqrt(2)*sigma)),p))+np.exp(-1.0*np.power(((r+Rgeo)/(np.sqrt(2)*sigma)),p)))
    return fgeo
    
    

def fit_gm(fluence,pos,Erad_gm,fit_params_geo):
   
    sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use=helper.return_sorted(fluence,pos)

    #fit_params_geo = Parameters()
    #fit_params_geo.add( 'Rgeo', value=160, min=-210,  max=1500)
    #fit_params_geo.add( 'sigma', value=90, min=10,  max=1500)

    fit_geo_x=sorted_pos_gm_use#[sorted_pos_gm_use>0]
    fit_geo_y=flu_gm_use#[sorted_pos_gm_use>0]

    rad_use=Erad_gm

    result = minimize(objective_gm, fit_params_geo, args=(fit_geo_x, fit_geo_y,rad_use,2))
    Rgeo_fit=result.params['Rgeo'].value
    sigma_fit=result.params['sigma'].value
    return Rgeo_fit,sigma_fit

    
###########################################################################################
    
    
def objective_gauss_sigmoid(params, r,data):
    resid_0 = 0.0*data[:]
    error=0.05*data+0.05*np.max(data)
    sigma=error*np.ones([len(data)])
    resid_0 = (data - return_gm_gauss_sigmoid_fit(params,r))**2/sigma**2

    return resid_0.flatten()
    


def return_gm_gauss_sigmoid_fit(params,r):
    r=np.abs(r)
    r0=params['r0'].value
    r02=params['r02'].value
    sigma=params['sigma'].value
    p0=params['p0'].value
    A=params['A'].value
    a_rel=params['a_rel'].value
    s=params['s'].value

    p=2*np.ones([len(r)])
        
    for i in np.arange(len(r)):
        if r[i]>r0:
            #p[i]=p0
            p[i]=2*np.power((r0/r[i]),(p0/1e3))

    return A*((np.exp(-1*((r-r0)/sigma)**p))+(a_rel/(1+np.exp(s*(r/(r0-r02))))))


def return_gm_gauss_sigmoid(r,A,sigma,r0,r02,p0,a_rel,s):
    r=np.abs(r)
    p=2*np.ones([len(r)])
        
    for i in np.arange(len(r)):
        if r[i]>=r0:
            #p[i]=p0
            p[i]=2*np.power((r0/r[i]),(p0/1e3))

        
    return A*((np.exp(-1*((r-r0)/sigma)**p))+(a_rel/(1+np.exp(s*(r/(r0-r02))))))
        
        
        
def fit_gm_gauss_sigmoid(fluence,pos,fit_params_geo):
   
    sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use=helper.return_sorted(fluence,pos)

    #fit_params_geo = Parameters()
    #fit_params_geo.add( 'A', value=2, min=.1,  max=15)
    #fit_params_geo.add( 'sigma', value=90, min=10,  max=1500)
    #fit_params_geo.add( 'r0', value=90, min=10,  max=1500)
    #fit_params_geo.add( 'r02', value=90, min=10,  max=1500)
    #fit_params_geo.add( 'p0', value=2, min=1,  max=3)
    #fit_params_geo.add( 'a_rel', value=.6, min=.4,  max=1.5)
     
    fit_geo_x=sorted_pos_gm_use#[sorted_pos_gm_use>0]
    fit_geo_y=flu_gm_use#[sorted_pos_gm_use>0]
    #rad_use=Erad_gm

    result = minimize(objective_gauss_sigmoid, fit_params_geo, args=(fit_geo_x, fit_geo_y))
    A_fit=result.params['A'].value
    sigma_fit=result.params['sigma'].value
    r0_fit=result.params['r0'].value
    r02_fit=result.params['r02'].value
    p0_fit=result.params['p0'].value
    a_rel_fit=result.params['a_rel'].value
    s_fit=result.params['s'].value

    return A_fit,sigma_fit,r0_fit,r02_fit,p0_fit,a_rel_fit,s_fit




def get_chi2(fluence, pos,A,sigma,r0,r02,p0,a_rel,s):

    sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use=helper.return_sorted(fluence,pos)
    
    resid_0 = 0.0*flu_gm_use[:]
    error=0.1*np.max(flu_gm_use)
    sigma=error*np.ones([len(flu_gm_use)])
    resid_0 = (flu_gm_use - return_gm_gauss_sigmoid(sorted_pos_gm_use,A,sigma,r0,r02,p0,a_rel,s))**2/sigma**2

    return np.sum(resid_0.flatten())/(len(resid_0.flatten())-6)

    




def do_fits(l):
    events=random.sample(range(nSims), l)

    events2=np.array([])

    energy2=np.array([])
    dmax2=np.array([])
    alpha2=np.array([])
    clip_ratio2=np.array([])
    cherenkov_angle2=np.array([])
    cherenkov_r2=np.array([])
    Erad_gm_2=np.array([])
    Erad_ce_2=np.array([])

    
    
    A=np.array([])
    sigma=np.array([])
    r0=np.array([])
    r02=np.array([])
    p0=np.array([])
    a_rel=np.array([])
    s=np.array([])
    chi2=np.array([])
    event2=np.array([])

    
    
    
    
    for n in np.arange(l):

        sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use=helper.return_sorted(fluence_50_350[n],ant_pos[n])
        fit_params_geo_sig = Parameters()
        fit_params_geo_sig.add( 'A', value=2., min=.01,  max=150.)
        fit_params_geo_sig.add( 'sigma', value=90., min=10.,  max=1500.)
        fit_params_geo_sig.add( 'r0', value=100., min=1.,  max=300.)

        fit_params_geo_sig.add( 'r02', value=80., min=1.,  max=300.)
        fit_params_geo_sig.add( 'p0', value=2., min=1.,  max=3.)

        fit_params_geo_sig.add( 'a_rel', value=1.5, min=-0.1,  max=3.)
        #fit_params_geo_sig.add( 's', value=4.5,min=0, max=6.)
        fit_params_geo_sig.add( 's', value=2.0,vary=False)

        A_fit,sigma_fit,r0_fit,r02_fit,p0_fit,a_rel_fit,s_fit= fit.fit_gm_gauss_sigmoid(fluence_50_350[n],ant_pos[n],fit_params_geo_sig)

        f_geo_sig=fit.return_gm_gauss_sigmoid(r_plot,A_fit,sigma_fit,r0_fit,r02_fit,p0_fit,a_rel_fit,s_fit)
        chi2_fit=fit.get_chi2(fluence_50_350[n],ant_pos[n],A_fit,sigma_fit,r0_fit,r02_fit,p0_fit,a_rel_fit,s_fit)
        
        events2= np.append(events2,[events[n]], axis=0)

        energy2 = np.append(energy2,[energy[n]], axis=0)
        dmax2 = np.append(dmax2,[dmax[n]], axis=0)
        alpha2 = np.append(alpha2,[alpha[n]], axis=0)
        clip_ratio2 = np.append(clip_ratio2,[clip_ratio[n]], axis=0)
        cherenkov_angle2 = np.append(cherenkov_angle2,[cherenkov_angle[n]], axis=0)
        cherenkov_r2 = np.append(cherenkov_r2,[cherenkov_r[n]], axis=0)
        A = np.append(A,[A_fit], axis=0)
        sigma = np.append(sigma,[sigma_fit], axis=0)
        r0 = np.append(r0,[r0_fit], axis=0)
        r02 = np.append(r02,[r02_fit], axis=0)
        p0 = np.append(p0,[p0_fit], axis=0)
        a_rel = np.append(a_rel,[a_rel_fit], axis=0)
        s = np.append(s,[s_fit], axis=0)
        chi2 = np.append(chi2,[chi2_fit], axis=0)
        Erad_gm_2=np.append(Erad_gm_2,[Erad_gm_50_350[n]], axis=0)
        Erad_ce_2=np.append(Erad_ce_2,[Erad_ce_50_350[n]], axis=0)


        '''
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
     
        ax1.plot(sorted_pos,flu_gm,'.',color='grey')
        ax1.plot(sorted_pos_gm_use,flu_gm_use,'.',color='black')
        ax1.plot(r_plot,f_geo_sig,color='green')
    
        ax2.plot(sorted_pos,flu_gm,'.',color='grey')
        ax2.plot(sorted_pos_gm_use,flu_gm_use,'.',color='black')
        ax2.plot(r_plot,f_geo_sig,color='green')

        ax2.set_yscale('log')

        ax1.grid()
        ax2.grid()

        plt.show()
        '''
    return events2,energy2,Erad_gm_2,Erad_ce_2,dmax2,alpha2,clip_ratio2,cherenkov_angle2,cherenkov_r2,A,sigma,r0,r02,p0,a_rel,s,chi2




