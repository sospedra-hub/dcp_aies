import torch
from torch.optim import lr_scheduler


class parameters(object):
    '''
    class to store parameters of neural network
    '''
    
    def __init__(self,
                 model):
        
        self.params = {
                    "model"               : model,
                    "hidden_dims"         : [[1500, 720, 360, 180, 90],
                                             [180, 360, 720, 1500]],
                    "time_features"       :  ['month_sin',
                                              'month_cos'],
                    "extra_predictors"    : None, 
                    "epochs"              : 60,
                    "batch_size"          : 64,
                    "batch_normalization" : False,
                    "dropout_rate"        : 0.2,
                    "L2_reg"              : 0,
                    "append_mode"         : 3,                  
                    "hyperparam"          : 1, 
                    "optimizer"           : torch.optim.Adam,
                    "lr"                  : 0.001 ,
                    "loss_region"         : None,
                    "subset_dimensions"   : None,
                    "lead_time_mask"      : None,
                    "lr_scheduler"        : True,
                    "start_factor"        : 1.0,
                    "end_factor"          : 0.1,
                    "total_iters"         : 50,  
                    "Biome"               : None,
                    "arch"                : 3,
                    "version"             : 3, ### 1 , 2 ,3
                 }

        
    def PrintDict(self):
        print(self.params)
            
