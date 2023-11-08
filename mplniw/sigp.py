# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:niskine]
#     language: python
#     name: conda-env-niskine-py
# ---

# %% [markdown]
# #### Imports

# %% janus={"all_versions_showing": false, "cell_hidden": false, "current_version": 0, "id": "80aa11a68a82c8", "named_versions": [], "output_hidden": false, "show_versions": false, "source_hidden": false, "versions": []}
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import gsw
import scipy.signal as signal
#import gvpy as gv
import niskine

min2d=1/(60*24)

def ni_bandpass_adcp(adcp, bandwidth=1.06):
    tlow, thigh = niskine.calcs.determine_ni_band(bandwidth=1.06)
    outu = adcp.u.copy()
    outu = outu.interpolate_na(dim="time", max_gap=np.timedelta64(8, "h"))
    outv = adcp.v.copy()
    outv = outv.interpolate_na(dim="time", max_gap=np.timedelta64(8, "h"))
    i = 0
    for g, aai in outu.groupby("z"):
        outu[i, :] = niskine.calcs.bandpass_time_series(aai.data, tlow, thigh, fs=6)
        i += 1
    i = 0
    for g, aai in outv.groupby("z"):
        outv[i, :] = niskine.calcs.bandpass_time_series(aai.data, tlow, thigh, fs=6)
        i += 1
    adcp["bpu"] = outu
    adcp["bpv"] = outv
    return adcp
    
def calc_ni_eke(adcp):
    rho = 1025
    # load WKB normalization matrix
    wkb = niskine.clim.get_wkb_factors(adcp)
    # calculate NI EKE
    adcp["ni_eke"] = 0.5 * rho * ((wkb * adcp.bpu) ** 2 + (wkb * adcp.bpv) ** 2)
    return adcp,wkb

def _get_E(x, ufunc=True, **kwargs):
    ax = -1 if ufunc else 0
    #
    dkwargs = {
        "window": "hann",
        "return_onesided": False,
        "detrend": None,
        "scaling": "density",
    }
    dkwargs.update(kwargs)
    f, E = signal.welch(x, fs=6*24.0, axis=ax, **dkwargs)
    #
    if ufunc:
        return E
    else:
        return f, E


def get_E(v, f=None, **kwargs):
    #v = v.chunk({"time": len(v.time)})
    if "nperseg" in kwargs:
        Nb = kwargs["nperseg"]
    else:
        Nb = 60 * 24
        kwargs["nperseg"] = Nb
    if "return_onesided" in kwargs and kwargs["return_onesided"]:
        Nb = int(Nb/2)+1
    if f is None:
        f, E = _get_E(v.values, ufunc=False, **kwargs)
        return f, E
    else:
        E = xr.apply_ufunc(
            _get_E,
            v,
            dask="parallelized",
            output_dtypes=[np.float64],
            input_core_dims=[["time"]],
            output_core_dims=[["freq_time"]],
            dask_gufunc_kwargs={"output_sizes": {"freq_time": Nb}},
            kwargs=kwargs,
        )
        E = E.assign_coords(freq_time=f).sortby("freq_time")
        E.attrs.update({'long_name':'PSD','units':r'$m^2.s^{-1}$'})
        E.freq_time.attrs.update({'long_name':'Frequency','units':'cpd'})
        return E

def wrap_spectra(ds, v, Nb=30*24*6,**kwargs):
    f, E = get_E(ds[v].isel(z=1), nperseg=Nb,**kwargs)
    E = get_E(ds[v], f=f, nperseg=Nb,detrend=False,**kwargs).compute()
    return f,E

def convolve(x, h=None, hilbert=False):
    """ Convolve an input signal with a kernel
    Optionaly compute the Hilbert transform of the resulting time series
    
    Parameters
    x : input signal
    h : filter 
    hilbert : True for Hilbert transform to be applied to the filtered signal
    
    Returns
    x_f : filtered signal or hilbert transform of the filtered signal
    """
    #x_c = np.nan_to_num(x,0)
    #argnan = np.where(np.isnan(x))
    x_c = x.copy()
    x_fill = np.nan_to_num(x_c)
    argnan = np.where(np.isnan(x_c)==True)
    x_f = signal.filtfilt(h, [1], x_fill, axis=-1,padlen=0)#
    x_f[argnan]=np.nan
    if hilbert:
        return signal.hilbert(x_f)
    else:
        return x_f

def filt(v, h, hilbert=False):
    
    output_dtype = complex if hilbert else float
    gufunc_kwargs = dict(output_sizes={'time': len(v.time)})
    return xr.apply_ufunc(convolve, v, kwargs={'h': h, 'hilbert': hilbert},
                    dask='parallelized', output_dtypes=[output_dtype],
                    input_core_dims=[['time']],
                    output_core_dims=[['time']],
                    dask_gufunc_kwargs = gufunc_kwargs,
                         )
#def hilbert_transform(v):
#    return signal.
def wrap_hilbert(ds,v):
    output_dtype = complex
    gufunc_kwargs = dict(output_sizes={'time': len(v.time)})
    ds_copy = ds.copy()
    for v in ['u','v']:
        ds_copy[v+'_filtered'] = xr.apply_ufunc(signalhilbert, ds[v].fillna(0), output_dtypes=[output_dtype],
                    input_core_dims=[['time']],
                    output_core_dims=[['time']],
                    dask_gufunc_kwargs = gufunc_kwargs,
                         )
    return ds_copy
    
def wrap_filter(ds,h,columns=['u','v'],**kwargs):
    ds_copy = ds.copy()
    for v in columns:
        ds_copy[v+'_filtered'] = filt(ds[v],h,**kwargs)
    return ds_copy

def interpolation(dict_array, z_new, t_new, minDT=60, dt=10, fill_gaps=True, **kwargs):
    """
    dict_xarray: Contains the adcp nuber as key and corresponding DataArray in item, dict
    z_new: regular depth grid to use for interpolation, numpy array
    t_new: regular time grid to use for interpolation, pandas date range
    fill_gaps: optional (default is True), Gaps in depth are filled using interpolation
    """
    _array = dict_array.copy()
    for a in dict_array:
        if dict_array[a].time.size<minDT/(dt*min2d):
            del _array[a]
    list_interp = [_array[a].sel(time=slice(t_new[0],t_new[-1])).interp(time=t_new,z=z_new) for a in _array]
    if len(list_interp)==1:
        ds = list_interp[0]
    elif len(list_interp)>1:
        ds = list_interp[0].copy()
        for l in list_interp[1:]:
            ds = ds.combine_first(l)
    else :
        raise Error
        
    if fill_gaps:
        return ds,ds.interpolate_na(dim='z',**kwargs)
    else:
        return ds