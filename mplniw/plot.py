import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_uv(ds,V,fig,ax,cmap = 'coolwarm', vmax=0.4,**kwargs):
    """
    Wrapper to plot 2D variables (Depth, Time)
    ds : xarray, Dataset
    V : Variable to be plotted
    fig,ax: figure and ax to use to plot the variable
    
    """
    ds[V].plot(ax=ax, yincrease=False,vmax=vmax,cmap=cmap, **kwargs)
    return fig,ax