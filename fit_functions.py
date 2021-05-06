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
import radiation_energy as radiation_energy
from scipy import special
import random
import scipy.interpolate as interpolate



def objective_v1(params, r, data,Egeo_prime,p):
    resid_0 = 0.0*data[:]
    error=0.1*np.max(data)
    sigma=error*np.ones([len(data)])
    resid_0 = (data - return_gm_fit(params,r,Egeo_prime,p))**2/sigma**2

    return resid_0.flatten()

def fit_geo(fluence,pos,Erad_gm):
   
    sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use=return_sorted(fluence,pos)

    fit_params_geo = Parameters()
    fit_params_geo.add( 'Rgeo', value=160, min=-210,  max=1500)
    fit_params_geo.add( 'sigma', value=90, min=10,  max=1500)

    fit_geo_x=sorted_pos_gm_use#[sorted_pos_gm_use>0]
    fit_geo_y=flu_gm_use#[sorted_pos_gm_use>0]

    rad_use=Erad_gm

    result = minimize(objective_v1, fit_params_geo, args=(fit_geo_x, fit_geo_y,rad_use,2))
    Rgeo_fit=result.params['Rgeo'].value
    sigma_fit=result.params['sigma'].value
    return Rgeo_fit,sigma_fit
