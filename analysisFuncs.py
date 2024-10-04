import xarray as xr
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.colors import ListedColormap

cmap = plt.get_cmap('jet')#'Blues'
my_cmap = cmap(np.arange(cmap.N)) # Get the colormap colors
my_cmap[:,-1] = 1 - np.flip(np.geomspace(0.01, 1, cmap.N, endpoint=False)) # Set alpha
my_cmap = ListedColormap(my_cmap) # Create new colormap


params2 = {
    'font.family': 'sans serif',
    'lines.markersize': 2,
    'lines.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.facecolor': 'whitesmoke',
    'axes.linewidth': 1.5,
    'legend.title_fontsize': 15,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.size': 8,'xtick.minor.size': 5,'xtick.major.width': 1.5,'xtick.minor.width': 1.,
    'ytick.major.size': 8,'ytick.minor.size': 5,'ytick.major.width': 1.5,'ytick.minor.width': 1.,
    'legend.frameon': False,
}

plt.rcParams.update(params2)

def label_plot(ax, proj, top=False,bottom=True,left=True,right=False):
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.OCEAN, color="white", edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.7)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.25, color='grey', alpha=1., linestyle='--')
    gl.top_labels = top
    gl.bottom_labels = bottom
    gl.right_labels = right
    gl.left_labels = left


def plot_map(da, ax=None, proj=None, yaxis="lat", xaxis="lon", cmap=None, vmax=None, vmin=None, colorbar=True, top=False,bottom=True,left=True,right=False,
             colorbar_label="Precipitation [mm]", xlim=None, ylim=None,alpha=1.0,norm=None):
    
    if proj is None: proj=ccrs.PlateCarree()
    if ax is None: ax = plt.axes(projection=proj)

    if colorbar == True: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin,alpha=alpha, add_colorbar=True,cbar_kwargs={'shrink':0.6})
    elif norm!=None: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, norm=norm, ax=ax, transform=proj, add_colorbar=False,alpha=alpha)
    else: qm = da.plot.pcolormesh(xaxis, yaxis, cmap=cmap, ax=ax, transform=proj, vmax=vmax, vmin=vmin, add_colorbar=False,alpha=alpha)
    label_plot(ax, proj, top=top, bottom=bottom, left=left, right=right)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if colorbar:
        qm.colorbar.set_label(colorbar_label)


# # -----------------------------------------------------------------

def getCDF(data):

    # use np.percentile to calculate empirical cdf
    x = np.append(np.arange(0.5,100.,0.5),[99.8,99.9])
    cdf = data.chunk(dict(time=-1)).quantile(x/100.)
    #cdf = np.array(list([np.percentile(data,i) for i in x]))
    
    return(x,cdf)


# # -----------------------------------------------------------------
# Function to find annual climate metric, func, from a dataset, ds

def applyFunc(ds,func,thr=0.,time_var='time'):
    
    if time_var!='time':
        ds = ds.rename({time_var:'time'})
    
    # # first check if variable seems to be in K and convert to C if so
    # if ds.mean()>250.:
    #     ds = ds - 273.15
    
    if func == 'std':
        result = ds.groupby('time.year').std(skipna=True)
    
    elif func == 'sum':
        result = ds.groupby('time.year').sum(skipna=True)
    
    elif func == 'mean':
        result = ds.groupby('time.year').mean(skipna=True)
    
    elif func == 'max':
        result = ds.groupby('time.year').max(skipna=True)
    
    elif func == 'q95':
        result = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.95)
    
    elif func == 'q99':
        result = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.99)
        
    elif func == 'count':
        result = ds.where(ds>thr).groupby('time.year').count()
    
    elif func == 'cwd':
        # find highest number of consecutive wet day spell each year
        t=3
        cwdmax = t
        windows = []
        while cwdmax==t:
            cwd = ds.where(ds>thr).rolling(time=t,center=True).count()
            cwd = cwd.groupby('time.year').max()
            windows.append(cwd)
            cwdmax = cwd.max()
            t+=1
            
        result = ds.where(ds>thr).rolling(time=30,center=True).count()
        result = result.groupby('time.year').max()
    
    elif func == 'cdd':
        # find highest number of consecutive dry day spell each year
        result = ds.where(ds==0.).rolling(time=90,center=True).count()
        result = result.groupby('time.year').max()
    
    elif func == 'r95ptot':
        if thr==0.:
            print('No threshold given. Calculating from ds.')
            thr = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.95)
            years = thr.year.values
            if len(years)>30:
                thr = thr.sel(year=slice(years[0],years[30])).mean('year')
            else: thr = thr.mean('year')
        
        annualtot = ds.groupby('time.year').sum().values
        result = ds.where(ds>thr).groupby('time.year').sum()/annualtot
    
    elif func == 'r99ptot':
        if thr==0.:
            print('No threshold given. Calculating from ds.')
            thr = ds.chunk(dict(time=-1)).groupby('time.year').quantile(0.99)
            years = thr.year.values
            thr = thr.sel(year=slice(years[0],years[30])).mean('year')
        
        annualtot = ds.groupby('time.year').sum().values
        result = ds.where(ds>thr).groupby('time.year').sum()/annualtot
    
    elif func == 'trend':
        result = ds.polyfit(dim="year", deg=1,skipna=True).polyfit_coefficients.sel(degree=1)
    
    else:
        print('This climate metric is not recognized')
    
    return(result)


# # -----------------------------------------------------------------

def getRegion(ds, coords,latname='lat',lonname='lon',time=0.,time_var='time'):

    if time_var!='time':
        ds = ds.rename({time_var:'time'})
    
    if latname!='lat':
        ds = ds.rename({latname:'lat'})
        
    if lonname!='lon':
        ds = ds.rename({lonname:'lon'})
    
    ds = ds.reindex(lat=np.sort(ds.lat))
    
    if len(coords)==2:
        
        if ds.lon.max().values>180.:
            dsnew = ds.sel(lat=coords[0],lon=coords[1]+360.,method='nearest')
        
        else:
            dsnew = ds.sel(lat=coords[0],lon=coords[1],method='nearest')
    
    elif len(coords)==4:
        
        if ds.lon.max().values>180.:
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[2]+360.,coords[3]+360.))
        
        elif ds.lon[0].values<ds.lon[-1].values:
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[2],coords[3]))
        
        else:
            print(ds.lon[0].values,ds.lon[-1].values)
            dsnew = ds.sel(lat=slice(coords[0],coords[1]),lon=slice(coords[3],coords[2]))
    
    else:
        print('Coordinate format not recognized')
    
    if time!=0.:
        dsnew = dsnew.sel(time=slice(time[0],time[1]))

    if time_var!='time':
        dsnew = dsnew.rename({'time':time_var})
            
    return(dsnew)

# # ----------------------------------------------------------
