import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature   
from matplotlib import ticker
from pathlib import Path


def new_cmap_with_white(cmap):
    '''
    Modify colormap to replace endcolor with white
    '''
    
    cmap_x = plt.cm.get_cmap(cmap)
    cmap_list = cmap_x(np.linspace(0, 1, 256))
    cmap_list[np.argmax(np.all(cmap_list[:, :3] >= [0.99, 0.99, 0.0], axis=1))] = [1, 1, 1, 1] 
    new_cmap = colors.LinearSegmentedColormap.from_list(f'{cmap}_white', cmap_list)

    return new_cmap
    


def plot_maps_wmo(ds,
                  lat=None,
                  lon=None,
                  central_longitude=180,
                  gridlines=False,
                  cmap='RdBu_r', # mpl.cm.RdBu_r,
                  vmin=-1.5,
                  vmax=1.5,
                  nvals=12,
                  vals=None, 
                  cbar=False,
                  cbar_label='',
                  cbar_extend='both',
                  ncols=2,
                  imshow=False,
                  titles=None,
                  figsize=None,
                  fig_dir=None,
                  fig_name=None,
                  show=False,
                  save=False,
                  **kwargs):

    if central_longitude == 0: # remove white line at central longitude
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
    
    fnt_size = 14
    mpl.rcParams.update({'font.size': fnt_size})

    if lat is None:
        lat = ds.lat
    if lon is None:
        lon = ds.lon
        
    plt.close()

    fig, ax = plt.subplots(nrows=1,
                           ncols=len(titles), #len(ds.time),
                           figsize=figsize, 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)})

    if vals is None:
        vals = vmin + (vmax-vmin)*np.arange(nvals+1)/float(nvals)
        
    norm = mpl.colors.BoundaryNorm(vals,
                                   plt.cm.get_cmap(cmap).N)
    
    for i, (axis, ds_model) in enumerate(zip(ax, ds)):
        if gridlines:
            axis.gridlines(draw_labels=False)
            
        if imshow:
            img_extent = [lon[0], lon[-1], lat[0], lat[-1]]
            im = axis.imshow(ds_model, 
                             origin='lower',
                             extent=img_extent,
                             cmap=plt.cm.get_cmap(cmap),
                             norm=norm,
                             interpolation='none',                     
                             transform=ccrs.PlateCarree())
            im.set_clim(vmin,
                        vmax)
        else:  
            if vals is None:
                vals = np.linspace(vmin,
                                   vmax,
                                   nvals+1)

            fill = ds_model.plot.contourf(ax=axis,
                                          levels=vals,
                                          cmap=plt.cm.get_cmap(cmap),
                                          add_colorbar=False,
                                          transform=ccrs.PlateCarree())
        
        axis.add_feature(cfeature.NaturalEarthFeature('physical',
                                                      'land', 
                                                      '50m',
                                                      # edgecolor='face',
                                                      facecolor='gray'))            
        axis.coastlines()
        if titles:
            axis.set_title(titles[i],
                           fontsize=fnt_size)
    
    if cbar:
        if ncols == 2:
            clb_x = 0.2 
            clb_y = -0.1
            clb_w = 0.6
            clb_h = 0.07
        if ncols == 3:
            clb_x = 0.2 
            clb_y = 0.05 
            clb_w = 0.6
            clb_h = 0.05 
        if ncols == 4:
            clb_x = 0.2 
            clb_y = -0.1 
            clb_w = 0.6
            clb_h = 0.05 
        cax = plt.axes([clb_x, # left
                        clb_y, # bottom
                        clb_w, # width
                        clb_h])# height
        cb = mpl.colorbar.ColorbarBase(ax=cax,
                                       cmap=cmap,
                                       norm=norm,
                                       spacing='uniform',
                                       orientation='horizontal',
                                       extend=cbar_extend,
                                       ticks=vals)
        cb.set_label(label=cbar_label) #, **hfont) 
    fig.tight_layout()
    if save:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{fig_dir}/{fig_name}',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()   
    else:
        plt.close()



def plot_single_map_wmo(ds,
                        lat=None,
                        lon=None,
                        polar_stereo=False, 
                        central_longitude=180,
                        lat_lims=[50,90],
                        gridlines=False,
                        cmap=mpl.cm.RdYlBu,
                        vmin=-1.5,
                        vmax=1.5,
                        vals=None,
                        vals_signfig=2,                         
                        nvals=12,
                        cbar=False,
                        cbar_label='',
                        cbar_integer=False,
                        cbar_extend='both',
                        ticks_rotation=0,
                        ticks_centered=False,
                        title=None,
                        fnt_size=12,
                        imshow=False,
                        figsize=None,
                        fig_dir=None,
                        fig_name=None,
                        show=False,
                        save=False,
                        **kwargs): 

    mpl.rcParams.update({'font.size': fnt_size})

    if central_longitude == 0: # remove white line at central longitude
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
            
    if lat is None:
        lat = ds.lat
    if lon is None:
        lon = ds.lon

    crs = ccrs.PlateCarree(central_longitude=central_longitude)    
    if polar_stereo:
        crs = ccrs.NorthPolarStereo()    
        if max(lat_lims) < 0:
            crs = ccrs.SouthPolarStereo()    
        
    plt.close()
    
    fig, ax = plt.subplots(nrows=1,
                           ncols=1, 
                           figsize=figsize, 
                           subplot_kw={'projection' : crs})                           
    

    if vals is None:
        vals = vmin + (vmax-vmin)*np.arange(nvals+1)/nvals
    else:
        nvals = len(vals) - 1

        
    norm = mpl.colors.BoundaryNorm(vals,
                                   plt.cm.get_cmap(cmap).N)
    
    axis = ax
    if gridlines:
        axis.gridlines(draw_labels=False)
    if polar_stereo:
        polarCentral_set_latlim(lat_lims, axis)
        axis.add_feature(cfeature.NaturalEarthFeature('physical',
                                                      'land', 
                                                      '50m',
                                                      # edgecolor='face',
                                                      facecolor='grey'))
        
    if imshow:
        img_extent = [lon[0], lon[-1], lat[0], lat[-1]]
        im = axis.imshow(ds, 
                         origin='lower',
                         extent=img_extent,
                         cmap=plt.cm.get_cmap(cmap),
                         norm=norm,
                         interpolation='none',                     
                         transform=ccrs.PlateCarree())
    else:  

        if vals is None:
            vals = np.linspace(vmin,
                               vmax,
                               nvals+1)
        
        fill = ds.plot.contourf(ax=axis,
                                # levels=clevs,
                                levels=vals,
                                cmap=plt.cm.get_cmap(cmap),
                                add_colorbar=False,
                                transform=ccrs.PlateCarree())

    axis.add_feature(cfeature.NaturalEarthFeature('physical',
                                                  'land', 
                                                  '50m',
                                                  # edgecolor='face',
                                                  facecolor='gray'))            
    axis.coastlines()
    axis.set_title(title,
                   fontsize=fnt_size)

    if cbar:
        clb_x = 0.055 #0.095 
        clb_y = 0.1
        clb_w = 0.9 #0.8
        clb_h = 0.04
        if polar_stereo:
            clb_y = 0.0
        cax = plt.axes([clb_x, # left
                        clb_y, # bottom
                        clb_w, # width
                        clb_h])# height
        cb = mpl.colorbar.ColorbarBase(ax=cax,
                                       cmap=plt.cm.get_cmap(cmap),
                                       # cmap=cmap,
                                       norm=norm,
                                       spacing='uniform',
                                       orientation='horizontal',
                                       extend=cbar_extend,
                                       ticks=np.round(vals,
                                                      vals_signfig))
        cb.ax.tick_params(rotation=ticks_rotation)
        cb.set_label(label=cbar_label,
                     size=fnt_size-2) 
       
    fig.tight_layout()
    if save:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{fig_dir}/{fig_name}',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()   
    else:
        plt.ioff()


