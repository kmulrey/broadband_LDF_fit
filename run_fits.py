import numpy as np
import os
import imp
import pickle
import matplotlib.pyplot as plt
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
import sys
#sys.path.insert(0, "broadband_LDF_fit/")
import sim_helper as helper
import fit_functions as fit
import random
from optparse import OptionParser



parser = OptionParser()
parser.add_option("-e", "--event", default = "0", help = "event  index")
(options, args) = parser.parse_args()
e = int(options.event)
n=10*e

directory='/vol/astro3/lofar/sim/kmulrey/spectral_analysis/Srd_Data/'
files=glob.glob(directory+'*.p')

energy=np.array([])
zenith=np.array([])
azimuth=np.array([])
xmax=np.array([])
hi=np.array([])
rho=np.array([])
rho2=np.array([])
dmax=np.array([])
n_xmax=np.array([])
alpha=np.array([])
clip_ratio=np.array([])
Erad_30_80=np.array([])
Erad_gm_30_80=np.array([])
Erad_ce_30_80=np.array([])
Erad_30_200=np.array([])
Erad_gm_30_200=np.array([])
Erad_ce_30_200=np.array([])
Erad_50_350=np.array([])
Erad_gm_50_350=np.array([])
Erad_ce_50_350=np.array([])
em_dep=np.array([])
total_dep=np.array([])
prim=np.array([])
event=np.array([])
dmax_grams=np.array([])
cherenkov_angle=np.array([])
cherenkov_r=np.array([])
fluence_30_80=np.empty([0,160,2])
fluence_30_200=np.empty([0,160,2])
fluence_50_350=np.empty([0,160,2])
ant_pos=np.empty([0,160,3])


for i in np.arange(len(files)):
#for i in np.arange(2):
    event_no=int(files[i].split('/')[8].split('.')[0])
    infile=open(files[i],'rb')
    info=pickle.load(infile,encoding='latin1')
    infile.close()
    ratio=info['Erad_ce_30_80']/info['Erad_gm_30_80']
    if (len(ratio[ratio>1])>-1) and event_no!=73889001 and len(info['ant_pos'])>1 and (len(info['dmax_grams']) == len(info['zenith'])):
        n=len(info['energy'])
        e=np.ones([n])
        energy = np.append(energy,1e9*info['energy'], axis=0)
        zenith = np.append(zenith,info['zenith'], axis=0)
        azimuth = np.append(azimuth,info['azimuth'], axis=0)
        xmax = np.append(xmax,info['xmax'], axis=0)
        prim = np.append(prim,info['prim'], axis=0)
        hi = np.append(hi,info['hi'], axis=0)
        rho = np.append(rho,info['rho'], axis=0)
        rho2 = np.append(rho2,info['rho2'], axis=0)
        dmax = np.append(dmax,info['dmax'], axis=0)
        n_xmax = np.append(n_xmax,info['n_xmax'], axis=0)
        alpha = np.append(alpha,info['alpha'], axis=0)
        clip_ratio = np.append(clip_ratio,info['clip_ratio'], axis=0)
        Erad_30_80 = np.append(Erad_30_80,info['Erad_30_80'], axis=0)
        Erad_gm_30_80 = np.append(Erad_gm_30_80,info['Erad_gm_30_80'], axis=0)
        Erad_ce_30_80 = np.append(Erad_ce_30_80,info['Erad_ce_30_80'], axis=0)
        Erad_30_200 = np.append(Erad_30_200,info['Erad_30_200'], axis=0)
        Erad_gm_30_200 = np.append(Erad_gm_30_200,info['Erad_gm_30_200'], axis=0)
        Erad_ce_30_200 = np.append(Erad_ce_30_200,info['Erad_ce_30_200'], axis=0)
        Erad_50_350 = np.append(Erad_50_350,info['Erad_50_350'], axis=0)
        Erad_gm_50_350 = np.append(Erad_gm_50_350,info['Erad_gm_50_350'], axis=0)
        Erad_ce_50_350 = np.append(Erad_ce_50_350,info['Erad_ce_50_350'], axis=0)
        em_dep = np.append(em_dep,1e9*info['em_dep'], axis=0)
        total_dep = np.append(total_dep,1e9*info['total_dep'], axis=0)
        dmax_grams = np.append(dmax_grams,info['dmax_grams'], axis=0)
        cherenkov_angle = np.append(cherenkov_angle,info['cherenkov_angle'], axis=0)
        cherenkov_r = np.append(cherenkov_r,info['cherenkov_r'], axis=0)
        flu_30_80=np.asarray(info['fluence_30_80'])
        flu_30_200=np.asarray(info['fluence_30_200'])
        flu_50_350=np.asarray(info['fluence_50_350'])
        ant=np.asarray(info['ant_pos'])

        fluence_30_80=np.concatenate((fluence_30_80,flu_30_80), axis=0)
        fluence_30_200=np.concatenate((fluence_30_200,flu_30_200), axis=0)
        fluence_50_350=np.concatenate((fluence_50_350,flu_50_350), axis=0)
        ant_pos=np.concatenate((ant_pos,ant), axis=0)

        event=np.append(event,e*event_no, axis=0)




A_fit,sigma_fit,r0_fit,r02_fit,p0_fit,a_rel_fit,s_fit,chi2_fit=fit.do_fit(fluence_50_350[n],ant_pos[n])

info={'A_fit':A_fit,'sigma_fit':sigma_fit,'r0_fit':r0_fit,'r02_fit':r02_fit,'p0_fit':p0_fit,'a_rel_fit':a_rel_fit,'s_fit':s_fit,'chi2_fit':chi2_fit,'event':event[n],'energy':energy[n],'dmax':dmax[n],'alpha':alpha[n],'clip_ratio':clip_ratio[n],'cherenkov_angle':cherenkov_angle[n],'cherenkov_r':cherenkov_r[n],'Erad_gm_50_350':Erad_gm_50_350[n],'Erad_ce_50_350':Erad_ce_50_350[n]}



outfilename='/vol/astro7/lofar/kmulrey/energy_reco/fit_results_v1/'+str(int(event[n]))+'_'+str(e)+'.p'
outfile=open(outfilename,'wb')
pickle.dump(info,outfile)
outfile.close()

