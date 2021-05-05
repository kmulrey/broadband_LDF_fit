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



def return_sorted(fluence,pos):
    pos_vxvxb=pos[2::8]
    neg_vxvxb=pos[6::8]
    fluence_vxvxb_0=np.concatenate([fluence[2::8].T[0],fluence[6::8].T[0]])
    fluence_vxvxb_1=np.concatenate([fluence[2::8].T[1],fluence[6::8].T[1]])
    pos_vxvxb_all=np.concatenate([neg_vxvxb.T[1],pos_vxvxb.T[1]])
    
    inds = pos_vxvxb_all.argsort()
    sorted_pos=pos_vxvxb_all[inds]
    flu_gm=fluence_vxvxb_0[inds]
    flu_ce=fluence_vxvxb_1[inds]
    flu_gm_use=flu_gm[flu_gm>(1e-4*np.max(flu_gm))]
    flu_ce_use=flu_ce[flu_ce>(1e-4*np.max(flu_gm))]
    sorted_pos_gm_use=sorted_pos[flu_gm>(1e-4*np.max(flu_gm))]
    sorted_pos_ce_use=sorted_pos[flu_ce>(1e-4*np.max(flu_gm))]
    return sorted_pos,flu_gm,flu_ce,sorted_pos_gm_use,sorted_pos_ce_use,flu_gm_use,flu_ce_use
