import numpy as np
import os as os
import configparser
from os import path
import pandas as pd
import file_service as fs
import config_util as cu
import tensor_component_util as tcu
from tensortools.operations import unfold as tt_unfold, khatri_rao

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config


def dir_up(path,n): 
    for _ in range(n):
        path = dir_up(path.rpartition("\\")[0], 0) 
                                                       
    return(path)
        
def reconstruct_component_by_condition(cond, components_folder, condition_root_folder): 
    cond_name = cu.get_condition_name(cond)
    print("Components folder: " + str(components_folder))
    
    ref_fif_folder = os.path.join(condition_root_folder, 'fif')
    subject_comp =  tcu.get_component(components_folder, 1)
    print("subject_comp" + str(subject_comp))
    component_cnt = subject_comp.shape[1]
    
    fact_list = []
    for i in range(component_cnt):
        fact_name = "fact" + str(i+1)
        fact_list.append(fact_name)
        
    subject_comp_df = pd.DataFrame(subject_comp,columns = fact_list)
    subject_comp_df['k'] = np.arange(len(subject_comp_df))
    subject_fig_id = os.path.join(components_folder, 'subject_loadings.csv')
    subject_comp_df.to_csv(subject_fig_id, index = False)
    print("Subject Component Estimate Shape" + str(subject_comp.shape))
    
    temporal_comp =  tcu.get_component(components_folder, 2)
    print("Temporal Component Estimate Shape" + str(temporal_comp.shape))
    
    spatial_comp =  tcu.get_component(components_folder, 3)
    print("Spatial Component Estimate Shape" + str(spatial_comp.shape))
    
    x_reconctructed = tcu.get_reconstructed_component(components_folder)
    print("Reconstructed Estimate Shape" + str(x_reconctructed.shape))
    x_a = tt_unfold(x_reconctructed,mode=0)
    print("X A Shape" + str(x_a.shape))
    x_b = tt_unfold(x_reconctructed,mode=1)
    print("X B Shape" + str(x_b.shape))
    x_c = tt_unfold(x_reconctructed,mode=2)
    print("X C Estimate Shape" + str(x_c.shape)) 
    evoked_ref_path = os.path.join(ref_fif_folder, cond_name + '.fif')
    
    comp_cnt = subject_comp.shape[1]
    cnt = 0
    for t in range(comp_cnt):
        target_folder = os.path.join(components_folder, str(cnt+1))
        fs.ensure_dir(target_folder)
        print("Processing condition " + str(cond_name) + "; Target folder " + str(target_folder) + "; Component # " + str(cnt + 1))
        tcu.reconstruct_component(evoked_ref_path, subject_comp, temporal_comp, spatial_comp, target_folder, cnt, cond, temporal_comp, components_folder, condition_root_folder)
        cnt =  cnt + 1
