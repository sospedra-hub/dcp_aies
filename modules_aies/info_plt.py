import numpy as np


class plt_module(object):
    '''
    store information for plots
    '''

    def __init__(self,
                 data_list=None,
                 units=None,
                 color_dict=None):
        

        if color_dict is None:
            
            color_dict = {
                  data_list[0] : {
                                  'obs':'k',
                                  'raw':'tab:blue',
                                  'badj':'tab:cyan',
                                  'tadj':'tab:red',
                                  'nadj':'tab:purple'
                                  },
                         }
            if len(data_list) >= 2:
                color_dict = color_dict | {
                                           data_list[1] : {
                                                              'obs':'gray',
                                                              'raw':'tab:blue',
                                                              'badj':'tab:cyan',
                                                              'tadj':'tab:red',
                                                              'nadj':'tab:purple'
                                                           },
                                           }
            if len(data_list) >= 3:
                color_dict = color_dict | {
                                            data_list[2] : {
                                                              'obs':'silver',
                                                              'raw':'tab:orange',
                                                              'badj':'tab:green',
                                                              'tadj':'tab:olive',
                                                              'nadj':'tab:purple'
                                                               },
                                              }
            if len(data_list) >= 4:
                color_dict = color_dict | {
                                             data_list[3] : {
                                                              'obs':'lightgray',
                                                              'raw':'tab:purple',
                                                              'badj':'tab:brown',
                                                              'tadj':'tab:pink',
                                                              'nadj':'tab:purple'
                                                               },
                                              }
            if len(data_list) >= 5:
                for ii in np.arange(2,len(data_list)):
                    color_dict = color_dict | {
                                                 data_list[ii] : {
                                                              'obs':'whitesmoke',
                                                              'raw':'tab:purple',
                                                              'badj':'tab:brown',
                                                              'tadj':'tab:pink',
                                                              'nadj':'tab:purple'
                                                               },
                                              }
        
        self.plot_data_dict = {
                                'fig2' : {
                                          'ymin'  : -3.4,
                                          'ymax'  : 0.4,
                                          'ystep' : .1,
                                          'legend': { 
                                                      'ts1' : 'SOM-FNN',
                                                      'ts2' : 'Uninitalized',
                                                      'ts3' : 'Forecast',
                                                    },
                                          'title' : {
                                                      'hnd'      : '(a) Raw',
                                                      'hnd_badj' : '(b) Badj',
                                                      'hnd_tadj' : '(c) Tadj',
                                                      'hnd_nadj' : '(d) Nadj',
                                                     },
                                          },
                                'fig3' : {
                                           'clim'  : {
                                                      'ymin'       : -20.,
                                                      'ymax'       : 20.,
                                                      'vals'       : [-40,-20,-10,
                                                                      -5,-3,-1,
                                                                      0,1,3,5,
                                                                      10,20,40],                                                                                                                    'units'      : 'gC m-2 yr-1',
                                                      'cbar_label' : 'Climatology', 
                                                      'cmap'       : 'Spectral_r',
                                                      'title' : {
                                                                  'obs'      : '(a) SOM-FFN',
                                                                  'hnd'      : '(x) Raw',
                                                                  'hnd_badj' : '(b) Badj',
                                                                  'hnd_tadj' : '(c) Tadj',
                                                                  'hnd_nadj' : '(d) Nadj',
                                                                 },                                                                                                    },
                                           'bias'  : {
                                                      'ymin'       : -5.,
                                                      'ymax'       : 5.,
                                                      'vals'       : [-40,-20,-10,
                                                                      -5,-3,-1,
                                                                      0,1,3,5,
                                                                      10,20,40],                                                              
                                                      'units'      : 'gC m-2 yr-1',
                                                      'cbar_label' : 'Climatological Bias', 
                                                      # 'cmap'       : 'Spectral_r',
                                                      'cmap'       : 'RdBu_r',
                                                      'title' : {
                                                                  'obs'      : '(a) SOM-FFN',
                                                                  'hnd'      : '(x) Raw',
                                                                  'hnd_badj' : '(b) Badj',
                                                                  'hnd_tadj' : '(c) Tadj',
                                                                  'hnd_nadj' : '(d) Nadj',
                                                                 },                                                                                                   },
                                          },
                                'fig4' : {

                                           'mse' :
                                                   {
                                                     'ymin'        : .5, #.4,
                                                     'ymax'        : 3.3,
                                                     'ystep'       : .5,
                                                     'ylabel'      : f'RMSE ({units})', # given as squared root
                                                     'bbox'        : (.49,.5,.5,.5),
                                                     'show_leg'    : True,
                                                     'title'       : '(a)',
                                                   },
                                           'rmse' :
                                                   {
                                                     'ymin'        : .1,
                                                     'ymax'        : .7,
                                                     'ystep'       : .1,
                                                     'ylabel'      : f'RMSE ({units})', # given as squared root
                                                     'bbox'        : (.49,.5,.5,.5),
                                                     'show_leg'    : True,
                                                     'title'       : '(a)',
                                                   },                                    
                                           'corr': {
                                                     'ymin'        : 0.,
                                                     'ymax'        : .64, #.61,
                                                     'ystep'       : .1,
                                                     'ylabel'      : f'ACC',
                                                     'bbox'        : (.49,.5,.5,.5),
                                                     'show_leg'    : False,
                                                     'title'       : '(c)',
                                                   },
                                           'cord': {
                                                     'ymin'        : -.2, #-.25, #-.1,
                                                     'ymax'        : .44, #.41, #.44, #.41,
                                                     'ystep'       : .1,
                                                     'ylabel'      : f'ACC (detrended)',
                                                     'bbox'        : (.49,.5,.5,.5),
                                                     'show_leg'    : False,
                                                     'title'       : '(d)',                                               
                                                   },
                                           'patc': {
                                                     'ymin'        : .4,
                                                     'ymax'        : 1.01,
                                                     'ystep'       : .1,
                                                     'ylabel'      : f'Pattern Correlation',
                                                     'bbox'        : (.49,.5,.5,.5),
                                                     'show_leg'    : False,
                                                     'title'       : '(b)',                                               
                                                   },
                                          },
                                'fig5' : {
                                           'rmse' :
                                                   {
                                                     'cmap'        : 'Reds_r',
                                                     'nvals'       : 10,
                                                     'cbar_extend' : 'max',
                                                     'cbar_label'  : f'RMSE ({units})',
                                                     'show_leg'    : True,
                                                     'ymin'        : 0,
                                                     'ymax'        : 1.,
                                                     'title'       : {
                                                                      'hnd'      : '(a)',
                                                                      'hnd_badj' : '(d)',
                                                                      'hnd_tadj' : '(g)',
                                                                      'hnd_nadj' : '(j)',
                                                                     },
                                                   },
                                           'corr': {
                                                     'cmap'        : 'RdBu_r',
                                                     'nvals'       : 10,
                                                     'cbar_extend' : None,
                                                     'cbar_label'  : f'ACC',
                                                     'show_leg'    : True,
                                                     'ymin'        : -1,
                                                     'ymax'        : 1.,
                                                     'title'       : {
                                                                      'hnd'      : '(b)',
                                                                      'hnd_badj' : '(e)',
                                                                      'hnd_tadj' : '(h)',
                                                                      'hnd_nadj' : '(k)',
                                                                     },
                                                   },
                                           'cord': {
                                                     'cmap'        : 'RdBu_r',
                                                     'nvals'       : 10,
                                                     'cbar_extend' : None,
                                                     'cbar_label'  : f'ACC (detrended)',
                                                     'show_leg'    : True,
                                                     'ymin'        : -1,
                                                     'ymax'        : 1.,
                                                     'title'       : {
                                                                      'hnd'      : '(c)',
                                                                      'hnd_badj' : '(f)',
                                                                      'hnd_tadj' : '(i)',
                                                                      'hnd_nadj' : '(l)',
                                                                     },                                               
                                                   },
                                          },
                                'fig6' : {
                                           'ymin'  : 0.,
                                           'ymax'  : 0.03,
                                           'vals'  : [-0.04,-0.02,-0.01,
                                                      -0.005,-0.0025,-0.00125,
                                                      0,0.00125,0.0025,0.005,
                                                      0.01,0.02,0.04],
                                           'units' : 'KgC m-2 yr-1',
                                           'cmap'  : 'Spectral_r',
                                           'title' : {
                                                       'field' : {
                                                                  'obs'      : '(a) SOM-FFN',
                                                                  'hnd'      : '(b) Raw',
                                                                  'hnd_badj' : '(c) Badj',
                                                                  'hnd_nadj' : '(d) Nadj',
                                                                  },
                                                       'error' : {
                                                                  'hnd'      : '(x) Raw',
                                                                  'hnd_badj' : '(e) Badj',
                                                                  'hnd_nadj' : '(f) Nadj',
                                                                 }
                                                      }
                                         }
                               }
        
        for ii,data in enumerate(data_list):
            self.plot_data_dict['fig4'][data] = {
                                             'obs' : {
                                                     'label' : 'obs',
                                                     'color' : f"{color_dict[data]['obs']}",
                                                     'line_style'  : '-',
                                                     'line_width'  : 2.0,
                                                     'marker'      : 'o',
                                                     'marker_size' : 5,
                                                     },
                                             'obs_trnd' : {
                                                          'label' : '',
                                                          'color' : f"{color_dict[data]['obs']}",
                                                          'line_style' : '--',
                                                          'line_width' : 1.0,
                                                          },
                                             'fct'  : {
                                                      'label' : f"Raw",
                                                      'color' : f"{color_dict[data]['raw']}",
                                                      },
                                             'hnd'  : {
                                                      'label' : f"Raw",
                                                      # 'color' : f"{color_dict[data]['raw']}",
                                                      'color' : 'k',
                                                      'line_style' : '-',
                                                      'line_width' : 2.0,
                                                      'marker'     : 'o',
                                                      'marker_size' : 5,
                                                      },
                                             'hnd_trnd' : {
                                                      'label' : '',
                                                      'color' : f"{color_dict[data]['raw']}",
                                                      'line_style' : '--',
                                                      'line_width' : 1.0,
                                                        },
                                             'fct_badj' : {
                                                      'label' : f"Badj",
                                                      'color' : f"{color_dict[data]['badj']}",
                                                        },
                                             'hnd_badj' : {
                                                      'label' : f"Badj",
                                                      # 'color' : f"{color_dict[data]['badj']}",
                                                      'color' : 'tab:blue',
                                                      'line_style' : '-',
                                                      'line_width' : 2.0,
                                                      'marker'     : '^',
                                                      'marker_size' : 5,
                                                          },
                                             'hnd_badj_trnd' : {
                                                      'label' : '',
                                                      'color' : f"{color_dict[data]['badj']}",
                                                      'line_style' : '--',
                                                      'line_width' : 1.0,
                                                              },
                                             'fct_tadj' : {
                                                      'label' : f"Tadj",
                                                      'color' : f"{color_dict[data]['tadj']}",
                                                          },
                                             'hnd_tadj' : {
                                                      'label' : f"Tadj",
                                                      # 'color' : f"{color_dict[data]['tadj']}",
                                                      'color' : 'tab:red',
                                                      'line_style' : '-',
                                                      'line_width' : 2.0,
                                                      'marker'     : 'D',
                                                      'marker_size' : 5,
                                                          },
                                             'hnd_tadj_trnd' : {
                                                      'label' : '',
                                                      'color' : f"{color_dict[data]['tadj']}",
                                                      'line_style' : '--',
                                                      'line_width' : 1.0,
                                                               },
                                             'fct_nadj' : {
                                                      'label' : f"Nadj",
                                                      'color' : f"{color_dict[data]['nadj']}",
                                                          },
                                             'hnd_nadj' : {
                                                      'label' : f"Nadj",
                                                      # 'color' : f"{color_dict[data]['nadj']}",
                                                      'color' : 'tab:green',
                                                      'line_style' : '-',
                                                      'line_width' : 2.0,
                                                      'marker'     : 's',
                                                      'marker_size' : 5,
                                                          },
                                             'hnd_nadj_trnd' : {
                                                      'label' : '',
                                                      'color' : f"{color_dict[data]['nadj']}",
                                                      'line_style' : '--',
                                                      'line_width' : 1.0,
                                                               },
                                   }        

