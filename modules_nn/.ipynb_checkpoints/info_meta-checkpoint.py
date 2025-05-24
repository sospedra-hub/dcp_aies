class meta_module(object):
    '''
    Class Metadata
    Dictionaries for variables, coordinates, and global attributes
    Note: keys starting with * are not included in var/dim attrs of copied files
    '''
    

    def __init__(self):
        
        
        self.var_dict = { 
                         'tas'   : {
                                    '*realm' : 'Amon',
                                    '*grid'  : 'gn',
                                    'standard_name' : 'air_temperature',
                                    'long_name' : 'Near-Surface Air Temperature',
                                    'comment' : 'ST+273.16, DCPP_table_comment: near-surface (usually, 2 meter) air temperature',                             
                                    'units' : 'K',
                                    'original_name' : 'ST',                             
                                    'cell_methods' : 'area: time: mean',
                                    'cell_measures' : 'area: areacella',
                                    'missing_value' : 1.e+20,
                                     '_FillValue' : 1.e+20,
                                  },   
                         'ts'     : {
                                    '*realm' : 'Amon',
                                    '*grid'  : 'gn',
                                   'standard_name' : 'surface_temperature',
                                   'long_name'     : 'Surface Temperature', 
                                   'comment'       : 'GT+273.16, DCPP_table_comment: Temperature of the lower boundary of the atmosphere',
                                   'units' : 'K',
                                   'original_name' : 'GT',                             
                                   'cell_methods' : 'area: time: mean',
                                   'cell_measures' : 'area: areacella',
                                   'missing_value' : 1.e+20,
                                    '_FillValue' : 1.e+20,
                                    },
                         'pr'   : {
                                    '*realm' : 'Amon',
                                    '*grid'  : 'gn',
                                    'standard_name' : 'precipitation_flux',
                                    'long_name' : 'Precipitation',
                                    'comment' : 'includes both liquid and solid phases',
                                    'units' : 'kg m-2 s-1',
                                    'original_name' : 'PCP',
                                    'cell_methods' : 'area: time: mean',
                                    'cell_measures' : 'area: areacella',
                                    'missing_value' : 1.e+20,
                                     '_FillValue' : 1.e+20,
                                  },     
                         'psl'   : {
                                    '*realm' : 'Amon',
                                    '*grid'  : 'gn',
                                    'standard_name' : 'air_pressure_at_mean_sea_level',
                                    'long_name' : 'Sea Level Pressure',
                                    'comment' : 'PMSL*100, DCPP_table_comment: Sea Level Pressure',
                                    'units' : 'Pa',
                                    'original_name' : 'PMSL',
                                    'cell_methods' : 'area: time: mean',
                                    'cell_measures' : 'area: areacella',
                                    'missing_value' : 1.e+20,
                                     '_FillValue' : 1.e+20,
                                  },     
                         'fgco2' : {
                                    '*realm' : 'Omon',
                                    '*grid'  : 'gn',
                                    'standard_name' : '...',
                                    'long_name' : '...',
                                    'comment' : '...',
                                    'units' : '...',
                                    'original_name' : '...',
                                    'cell_methods' : 'area: time: mean',
                                    'cell_measures' : 'area: areacella',
                                    'missing_value' : 1.e+20,
                                     '_FillValue' : 1.e+20,
                                  },     
                        }
        
        self.coord_dict = {              
                           'time' : {
                                      'long_name'     : 'Time',
                                      'standard_name' : 'time',
                                    },
                           'lat' : {
                                      'long_name'     : 'Latitude',
                                      'standard_name' : 'latitude',
                                      'units'         : 'degrees_north',
                                    },
                           'lon' : {
                                      'long_name'     : 'Longitude',
                                      'standard_name' : 'longitude',
                                      'units'         : 'degrees_east',
                                    },
                           'height' : {
                                      'standard_name' : 'height',
                                      'units'         : 'm',
                                    },
                           'realization' : {
                                      'standard_name' : 'realization',
                                      'units'         : '1',
                                    },
                         }
        
        
    def global_attributes(self,
                          var=None,
                          year=None,
                          realization=None,
                          comment='unadjusted monthly values'):       
        
        if year < 2020:
            dcppx = 'dcppA-hindcast' 
        else:
            dcppx = 'dcppB-forecast'
            
        if realization is None:
            srealization = None
            xrealization = None
        else:
            if realization < 10:
                srealization = f'00{realization}'
            else:
                if realization < 100:
                    srealization = f'0{realization}'
                else:
                    srealization = f'{realization}'
            xrealization = srealization[1:]
        
        global_attr = {
  'CMIP5' : {
      'institution'   : 'CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)',
      'institute_id'  : 'CCCma',
      'experiment_id' : 'decadal1960',
      'experiment'    : '10-year run initialized in year 1960',
      'source'        : 'CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7',
      'model_id'      : 'CanCM4',
      'forcing'       : 'GHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)',
      'parent_experiment_id' : 'N/A',
      'parent_experiment_rip' : 'N/A',
      'parent_experiment' : 'N/A',
      'branch_time'       : '0.',
      'references'        : 'http://www.cccma.ec.gc.ca/models',
      'initialization_method' : '4',
      'physics_version'   : '1',
      'realization' : f'{realization}',
      'branch_time_YMDH' : f'{year}:01:01:00',
      'CCCma_runid' : f'DHFP1D_E{srealization}_I{year}_M01',
      'CCCma_parent_runid' : f'DHFP1_E{srealization}',
      'product' : 'output',
      'frequency' : 'monthly',
      'modeling_realm' : f"{self.var_dict[var]['*realm']}",
      'project_id' : 'CMIP5',
      'conventions' : 'CF-1.4',
      'title' : 'CanCM4 model output prepared for CMIP5 10- or 30-year run initialized in year 1960',
      'comment' : f'{comment}',                                 
      
                },
            
  'CMIP6' : {
    'CCCma_model_hash'   : 'Unknown',
    'CCCma_parent_runid' : f'd2a-asm-e{xrealization}',
    'CCCma_pycmor_hash'  : '13db8596c37129e414cad7ae31f2927ca8f5dd39',
    'CCCma_runid'        : 'd2a{year}196201e01',
    'CCCma_model_hash'   : 'Unknown',
    'CCCma_parent_runid' : f'd2a-asm-e{xrealization}',
    # 'CCCma_pycmor_hash'  : '13db8596c37129e414cad7ae31f2927ca8f5dd39',
    'CCCma_runid'        : f'd2a{year}01e{xrealization}',
    'Conventions'        : 'CF-1.7 CMIP-6.2',
    'YMDH_branch_time_in_child'  : f'{year}:01:01:00',
    'YMDH_branch_time_in_parent' : f'{year}:01:01:00',      
    'activity_id'                : 'DCPP',
    'branch_method'      : 'Spin-up documentation',
    # 'branch_time_in_child'       : '40880',
    # 'branch_time_in_parent'      : '40880.',
    'contact'            : 'ec.cccma.info-info.ccmac.ec@canada.ca',
    'creation_date'      : '2019-06-18T21:29:36Z',
    'data_specs_version' : '01.00.29',
    'experiment'         : 'hindcast initialized based on observations and using historical forcing',
    'experiment_id'      : f'{dcppx}',
    'external_variables' : 'areacella',
    'forcing_index'      : 1,
    'frequency'          : 'mon',
    'further_info_url'   : f'https://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.{dcppx}.s{year}.r{realization}i1p2f1',
    'grid'               : 'T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa',
    'grid_label'         : 'gn',
#                 :history = "2019-06-18T21:29:36Z ;rewrote data to be consistent with DCPP for variable tas found in table Amon." ;
    'initialization_index' : 1,
    'institution' : 'Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada, Victoria, BC V8P 5C2, Canada',
    'institution_id' : 'CCCma',
    'mip_era' : 'CMIP6', 
    # 'nominal_resolution' : '500 km',
    'parent_activity_id' : 'DCPP',
    'parent_experiment_id' : 'dcppA-assim', 
    'parent_mip_era'  : 'CMIP6',
    'parent_source_id' : 'CanESM5', 
    # 'parent_time_units' : 'days since 1850-01-01 0:0:0.0',
    'parent_variant_label' : f'r{realization}i1p2f1',
    'physics_index' : 2,
    'product' : 'model-output',
    'realization_index' : 1,
    # 'realm' : 'atmos', 
    'references' : 'Geophysical Model Development Special issue on CanESM5 (https://www.geosci-model-dev.net/special_issues.html)', 
    'source' : 'CanESM5 (2019): \n, aerosol: interactive\n, atmos: CanAM5 (T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa)\n, atmosChem: specified oxidants for aerosols\n, land: CLASS3.6/CTEM1.2\n, landIce: specified ice sheets\n, ocean: NEMO3.4.1 (ORCA1 tripolar grid, 1 deg with refinement to 1/3 deg within 20 degrees of the equator; 361 x 290 lon gitude/latitude; 45 vertical levels; top grid cell 0-6.19 m)\n, ocnBgchem: Canadian Model of Ocean Carbon (CMOC); NPZD ecosystem with OMIP prescribed carbonate chemistry\n, seaIce: LIM2',
    'source_id'  : 'CanESM5',
    # 'source_type' : 'AOGCM', 
    'sub_experiment'    : f'initialized near end of year {year-1}',
    'sub_experiment_id' : f's{year-1}',
    # 'table_id' : f'{realm}',
    # 'table_info'  : 'Creation Date:(20 February 2019) MD5:374fbe5a2bcca535c40f7f23da271e49', 
    'title' : 'CanESM5 output prepared for CMIP6',
    # 'tracking_id'   : 'hdl:21.14100/87d2f63a-04e0-495a-aab4-35c9099fe8de',
    'variable_id'   : f'{var}',
    'variant_label' : f'r{realization}i1p2f1',
    'version' : 'v20190429',
    'license' : 'CMIP6 model data produced by The Government of Canada (Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada) is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.',
    'cmor_version' : '3.4.0',      
             },    
            
                           }
        
        return global_attr
        
                                     
                                     
                            
