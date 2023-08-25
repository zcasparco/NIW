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

def plot_event2D(ds,v,mld,ti,tf,months,**kwargs):
    fig,axs = plt.subplots(1,len(ti),figsize=(5*len(ti),6))
    for i in range(len(ti)):
        ds.sel(time=slice(ti[i],tf[i]))[v].plot(ax=axs[i],zorder=-1,y='z',yincrease=False,**kwargs)
        mld.sel(time=slice(ti[i],tf[i])).plot(ax=axs[i],x='time',c='c',zorder=1)
        #axs[i].plot(mld.sel(time=slice(ti[i],tf[i])),c='c',lw=2,zorder=1)
        axs[i].set_title('Month %s'%months[i])
    return fig,axs

