#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:23:42 2023

@author: dmilakov
"""

import numpy as np
# import jax.numpy as jnp
from fitsio import FITS
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

hdul=FITS('./mockspec.fits')
linelist = hdul[3].read()


#%%
# https://github.com/jkrogager/VoigtFit/blob/master/VoigtFit/funcs/voigt.py
l0 = 1215.6701
f  = 0.416400
gam = 6.265E8 

def H_Voigt(a, x):
    """Voigt Profile Approximation from T. Tepper-Garcia (2006, 2007)."""
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)
def Voigt(wl, l0, f, N, b, gam, z=0):
    """
    Calculate the optical depth Voigt profile.
    Parameters
    ----------
    wl : array_like, shape (N)
        Wavelength grid in Angstroms at which to evaluate the optical depth.
    l0 : float
        Rest frame transition wavelength in Angstroms.
    f : float
        Oscillator strength.
    N : float
        Column density in units of cm^-2.
    b : float
        Velocity width of the Voigt profile in cm/s.
    gam : float
        Radiation damping constant, or Einstein constant (A_ul)
    z : float
        The redshift of the observed wavelength grid `l`.
    Returns
    -------
    tau : array_like, shape (N)
        Optical depth array evaluated at the input grid wavelengths `l`.
    """
    # ==== PARAMETERS ==================

    c = 2.99792e10        # cm/s
    m_e = 9.1094e-28       # g
    e = 4.8032e-10        # cgs units

    # ==================================
    # Calculate Profile

    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)

    # dl_D = b/c*l0
    wl = wl/(z+1.)
    # x = (wl - l0)/dl_D + 0.000001
    x = (c / b) * (1. - l0/wl)

    tau = np.float64(C_a) * N * H_Voigt(a, x)
    return tau


#%%

    
c = 299792458
components = ['matter','lambda']#,'radiation','curvature']
w = {'matter':0, 'radiation':1./3, 'lambda':-1, 'curvature':-1/3}
def E(z,OM=0.3,OL=0.7):
    omega = {'matter':OM, 'lambda':OL, 'radiation':1.-OM-OL, 'curvature':0}
    sum_comp = np.sum([omega[i]*(1+z)**(3*(1+w[i])) for i in components],axis=0)    
    return np.sqrt(sum_comp)
def H(z,H0=70.,OM=0.3,OL=0.7):
    return H0*E(z,OM,OL)
def zdot(z_em,z_obs=0.,H0=70.,OM=0.3,OL=0.7,unit='1/year'):
    # per_second = 3.240779289444365e-20 # km/s/Mpc to 1/s
    per_year   = 1.022712165045695e-12  # km/s/Mpc to 1/year
    zdot_ = (1+z_em)*H(z_obs,H0,OM,OL) - H(z_em,H0,OM,OL)
    
    h0 = H0/70
    zdot_div_h0 = zdot_/h0
    # if unit == '1/year':
    return zdot_div_h0 * per_year
    # if unit == '1/s':
    #     return zdot_div_h0 * per_second

def vdot(z_em,z_obs=0.,H0=70.,OM=0.3,OL=0.7,unit='1/year'):
    
    zdot_per_year = zdot(z_em,z_obs,H0,OM,OL,'1./year')
    vdot_per_year = c*zdot_per_year/(1+z_em)
    
    return vdot_per_year

def get_z_drift(t,z_em,z_obs=0.,H0=0.7,OM=0.3,OL=1.):
    return t*zdot(z_em,z_obs,H0,OM,OL)

def line_model(wl,line_N,line_z,line_b):
    l0 = 1215.6701
    f  = 0.416400
    gam = 6.265E8   
    
    tau = Voigt(wl,l0,f,line_N,line_b*1e5,gam,line_z)
    model = np.exp(-tau) - 1.
    # plt.plot(wl,model)
    return model
#%%
def spectrum_after_t(t,linelist,zmin=2.9,zmax=3.2,z_step=5e-5,
                     z_obs=0.,H0=70.,OM=0.3,OL=0.7):
    z_list  = linelist['Z']
    # zmin = 2.9
    # zmax = 3.2
    cut = np.where((z_list>=zmin)&(z_list<=zmax))[0]
    
    # zmin    = np.min(z_list)
    # zmax    = np.max(z_list)
    zrange  = zmax-zmin
    # z_step  = 0.00005
    z_space = np.arange(zmin-0.01*zrange,zmax+0.01*zrange,z_step)
    
    W       = (1 + z_space) * 1215.6701
    Y       = np.ones_like(z_space)
    
    
    
    
    for line in linelist[cut]:
        log_N     = line['COLDEN']
        b         = line['B']
        z_initial = line['Z']
        z_drift   = get_z_drift(t,z_initial,z_obs,H0,OM,OL)
        z_final   = z_initial + z_drift
        
        model     = line_model(W,log_N,z_final,b)
        Y        += model
    
    return W,Y

def setup_plot_params():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelpad'] = 4.0
    mpl.rcParams['xtick.top'] =            True   # draw ticks on the top side
    mpl.rcParams['xtick.bottom'] =        True    # draw ticks on the bottom side
    mpl.rcParams['xtick.labeltop'] =      False   # draw label on the top
    mpl.rcParams['xtick.labelbottom'] =   True    # draw label on the bottom
    mpl.rcParams['xtick.major.size'] =     3.5     # major tick size in points
    mpl.rcParams['xtick.minor.size'] =     2       # minor tick size in points
    mpl.rcParams['xtick.major.width'] =    0.8     # major tick width in points
    mpl.rcParams['xtick.minor.width'] =    0.6     # minor tick width in points
    mpl.rcParams['xtick.major.pad'] =     3.5     # distance to major tick label in points
    mpl.rcParams['xtick.minor.pad'] =     3.4     # distance to the minor tick label in points
    mpl.rcParams['xtick.color'] =         "black"   # color of the tick labels
    mpl.rcParams['xtick.labelsize'] =      "medium"  # fontsize of the tick labels
    mpl.rcParams['xtick.direction'] =      "in"     # direction: {in, out, inout}
    mpl.rcParams['xtick.minor.visible'] =  True   # visibility of minor ticks on x-axis
    mpl.rcParams['xtick.major.top'] =      True    # draw x axis top major ticks
    mpl.rcParams['xtick.major.bottom'] =   True    # draw x axis bottom major ticks
    mpl.rcParams['xtick.minor.top'] =      True    # draw x axis top minor ticks
    mpl.rcParams['xtick.minor.bottom'] =   True    # draw x axis bottom minor ticks
    mpl.rcParams['xtick.alignment'] =     "center"  # alignment of xticks
    
    mpl.rcParams['ytick.left'] =           True    # draw ticks on the left side
    mpl.rcParams['ytick.right'] =          True   # draw ticks on the right side
    mpl.rcParams['ytick.labelleft'] =      True    # draw tick labels on the left side
    mpl.rcParams['ytick.labelright'] =     False   # draw tick labels on the right side
    mpl.rcParams['ytick.major.size'] =     3.5     # major tick size in points
    mpl.rcParams['ytick.minor.size'] =     2       # minor tick size in points
    mpl.rcParams['ytick.major.width'] =    0.8     # major tick width in points
    mpl.rcParams['ytick.minor.width'] =    0.6     # minor tick width in points
    mpl.rcParams['ytick.major.pad'] =      3.5     # distance to major tick label in points
    mpl.rcParams['ytick.minor.pad'] =      3.4     # distance to the minor tick label in points
    mpl.rcParams['ytick.color'] =          "black"   # color of the tick labels
    mpl.rcParams['ytick.labelsize'] =      "medium"  # fontsize of the tick labels
    mpl.rcParams['ytick.direction'] =      "in"      # direction: {in, out, inout}
    mpl.rcParams['ytick.minor.visible'] =  False   # visibility of minor ticks on y-axis
    mpl.rcParams['ytick.major.left'] =     True    # draw y axis left major ticks
    mpl.rcParams['ytick.major.right'] =    True    # draw y axis right major ticks
    mpl.rcParams['ytick.minor.left'] =     True    # draw y axis left minor ticks
    mpl.rcParams['ytick.minor.right'] =    True    # draw y axis right minor ticks
    mpl.rcParams['ytick.alignment'] =      "center_baseline"  # alignment of yticks

    return None

def get_figax(*args,**kwargs):
    _ = setup_plot_params()
    figure,ax = plt.subplots(1,1,figsize = (6,4),*args,**kwargs)
    ax.set_xlabel("Wavelength "+r"(${\rm \AA}$)")
    figure.text(0.80,0.03,'D. Milaković 2023',fontsize=8,horizontalalignment='left',
                transform=figure.transFigure)
    return figure, ax

def plot_spectral_ratio(t,linelist,zmin=2.9,zmax=3.2,z_step=5e-5,
                     z_obs=0.,H0=70.,OM=0.3,OL=0.7,plot_zero_time=False,
                     identifier=None,dirname='ratio',save=False):
    figure, ax = get_figax()
    W,Y = spectrum_after_t(t, linelist,zmin,zmax,z_step,z_obs,H0,OM,OL)
    W,Y0 = spectrum_after_t(0, linelist,zmin,zmax,z_step,z_obs,H0,OM,OL)
    ax.plot(W,Y/Y0-1,drawstyle='steps-mid',c='k',lw=1.5)
    
    ax.set_ylabel(r"Flux ratio $-$ 1")
    scinotate(ax, 'y',bracket='round')

    if t>10:
        ax.set_title(f"Elapsed time = {t:12,.0f} years")
    else:
        ax.set_title(f"Elapsed time = {t:12,.2f} years")
    figure.tight_layout()
    
    if save:
        basedir = './images/'
        figname = f"{identifier}.png"
        path = os.path.join(*[basedir,dirname,figname])
        figure.savefig(path,dpi=400)
    return
        
def plot_spectrum_at_t(t,linelist,zmin=2.9,zmax=3.2,z_step=5e-5,
                     z_obs=0.,H0=70.,OM=0.3,OL=0.7,plot_zero_time=False,
                     identifier=None,dirname='zdot',save=False):
    
    figure, ax = get_figax()
    W,Y = spectrum_after_t(t, linelist,zmin,zmax,z_step,z_obs,H0,OM,OL)
    
    ax.plot(W,Y,c='k')
    if plot_zero_time:
        W,Y_ = spectrum_after_t(0, linelist,zmin,zmax,z_step,z_obs,H0,OM,OL)
        ax.plot(W,Y_,lw=0.5,c='grey',zorder=0)
    ax.set_ylim(-0.05,1.05)
    ax.set_ylabel("Flux (normalised)")
    if t>10:
        ax.set_title(f"Elapsed time = {t:12,.0f} years")
    else:
        ax.set_title(f"Elapsed time = {t:12,.2f} years")
    figure.tight_layout()
    if save:
        basedir = './images/'
        figname = f"{identifier}.png"
        path = os.path.join(*[basedir,dirname,figname])
        figure.savefig(path,dpi=400)
    return

def scinotate(ax,axis,exp=None,dec=1,bracket='square'):
    '''
    Args:
    ----
    ax (matplotlib.Axes instance)
    axis (str) : 'x' or 'y'
    exp (int) : exponential
    dec (int) : number of decimal points
    bracket (str) : 'round' or 'square'
    '''
    from matplotlib import ticker

    axsc = getattr(ax,'{0}axis'.format(axis))
    
    braleft = '[' 
    brarigh = ']'
    if bracket == 'round':
        braleft = '('
        brarigh = ')'
    
    oldlbl = getattr(ax,'get_{0}label'.format(axis))()
    loc    = oldlbl.find(brarigh)
    axlim  = getattr(ax,'get_{0}lim'.format(axis))()
    exp    = exp if exp is not None else np.floor(np.log10(axlim[1]))
    axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
    print(oldlbl, loc)
    
    if loc > 0:
        newlbl = oldlbl[:loc] + \
            r' $\times 10^{{{exp:0.0f}}}${br}'.format(exp=exp,br=brarigh)
    else:
        newlbl = oldlbl + \
            r' {bl}$\times 10^{{{exp:.0f}}}${br}'.format(exp=exp,br=brarigh,bl=braleft)
    print(newlbl)
    set_lbl = getattr(ax,'set_{0}label'.format(axis))
    set_lbl(newlbl)
    return 
def sciformat(x,y,exp,dec):
    if x==0:
        return ('{num:.{width}f}'.format(num=x,width=dec))
    return ('{num:.{width}f}'.format(num=x/10**exp,width=dec))


def main(tmin,tmax,frames):
    wmin = 4920; zmin=wmin/l0 -1
    wmax = 4960; zmax=wmax/l0 - 1
    T = np.logspace(np.log10(tmin),np.log10(tmax),frames)
    
    for i in range(frames+1):
        if i==0:
            t = 0e0
        else:
            t = T[i-1]
        plot_spectrum_at_t(t,linelist,zmin,zmax,plot_zero_time=True,
                           identifier=f'{i:04d}',dirname='zdot',save=True)
