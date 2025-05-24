import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def plot_ts(da_list,
            data,
            error_bars=False,
            label_list=None,
            color_list=None,
            line_list=None,
            linew_list=None,
            marker_list=None,
            line='-',
            linew=2.0,
            marker='',
            markersize=5,
            x_offset=0,
            time_dim='year',
            title='',
            xlabel='x',
            ylabel='y',
            bbox=(.88,.5,.5,.5),
            xmin=None,
            xmax=None,
            xstep=1,
            xticks_labels=None,
            yticks_labels=None,
            ymin=None,
            ymax=None,
            ystep=.2,
            ncol_labels=1,
            hline=False,
            hline_value=0,
            fontsize=14,
            dir_name=None,
            file_name=None,
            show_leg=True,
            show=False,
            save=False):
    
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.close()
        
    if show or save:
        
        xx = [time for time in data[time_dim].values+x_offset]
        
        plt.figure(figsize=(6,3))
        
        plt.title(title,
                  fontsize=fontsize+3)
        
        if hline:
            plt.axhline(y=hline_value,
                        color='k',
                        linestyle='--',
                        linewidth=1.0)
        
        for ind,da in enumerate(da_list):
            
            if line_list is None:
                line_list = [line] * len(da_list)
            if linew_list is None:
                linew_list = [linew] * len(da_list)
            if marker_list is None:
                marker_list = [marker] * len(da_list)
            if label_list is None:
                label_list = np.arange(len(da_list))

            if error_bars:
                yy       = data.sel(legend=ind).isel(quantile=1)
                yy_error = data.sel(legend=ind).isel(quantile=2)-data.sel(legend=ind).isel(quantile=0)
                plt.plot(xx,
                         yy,
                         linewidth=linew_list[ind],
                         linestyle=line_list[ind],
                         marker=marker_list[ind],
                         markersize=markersize,
                         label=label_list[ind],
                         color=color_list[ind])
                plt.errorbar(xx,
                             yy,
                             yerr=yy_error,
                             fmt='none',
                             capsize=5,
                             label=None,
                             color=color_list[ind])
                
            else:
                plt.plot(xx,
                         data.sel(legend=ind),
                         linewidth=linew_list[ind],
                         linestyle=line_list[ind],
                         marker=marker_list[ind],
                         markersize=markersize,
                         label=label_list[ind],
                         color=color_list[ind])
        
        if xmin is None:
            xmin = min(xx)
        if xmax is None:
            xmax = max(xx)+1

        if ymin is None:
            ymin = int(data.min())
        if ymax is None:
            ymax = int(data.max())+1
             
        if xticks_labels is not None:
            xticks_lab = xticks_labels
        else:
            xticks_lab = np.arange(xmin,
                                   xmax,
                                   xstep)

        plt.xticks(np.arange(xmin,
                             xmax,
                             xstep),
                    xticks_lab)
        
        plt.yticks(np.arange(ymin,
                             ymax,
                             ystep))

        plt.xlabel(xlabel,
                   fontsize=fontsize)
        plt.ylabel(ylabel,
                   fontsize=fontsize)
        
        plt.ylim(ymin,
                 ymax)

        if show_leg:
            plt.legend(loc='best',
                       bbox_to_anchor=bbox,
                       # fontsize='small',
                       fontsize=fontsize,
                       handlelength=1,
                       ncol=ncol_labels,
                       frameon=False)   
                   
                
        if save:
            Path(dir_name).mkdir(parents=True,
                                 exist_ok=True)
            plt.savefig(f'{dir_name}/{file_name}.png',
                            bbox_inches='tight',
                            dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


