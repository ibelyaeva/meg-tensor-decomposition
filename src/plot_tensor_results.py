import mne
import numpy as np
import matplotlib as plt
import os
import configparser
from os import path
import file_service as fs
import config_util as cu
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams["legend.frameon"] = False

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config
    
def merge_evoked(evoked_list, cond, target_folder, ch_type='mag'):
    merged_evoked = mne.combine_evoked(evoked_list, weights= 'nave')
    condition_name = cu.get_condition_name(cond)
    merged_evoked_folder_fif = os.path.join(target_folder, 'fif')
    fs.ensure_dir(merged_evoked_folder_fif)

    merged_evoked_folder_fig = os.path.join(target_folder, 'fig')
    fs.ensure_dir(merged_evoked_folder_fig)
    
    file_name = condition_name + '.fif'
    merged_evoked_file_id = os.path.join(merged_evoked_folder_fif, file_name)
    print("merged_evoked shape= " + str(merged_evoked.data.shape))
    print("Writing Merged Evoked for Condition =" + str(condition_name) + "; @ " + str(merged_evoked_file_id))
    mne.write_evokeds(merged_evoked_file_id, merged_evoked)
    
    file_np_name = condition_name + '.npy'
    merged_evoked_np_id = os.path.join(merged_evoked_folder_fif, file_np_name)
    np.save(merged_evoked_np_id, merged_evoked.data)
    
    print("Merged Evoked Channels: " + str(merged_evoked.info['ch_names']))
    merged_evoked_plot_fig_id = os.path.join(merged_evoked_folder_fig, condition_name +  '_' + 'merged_evoked.pdf') 
    merged_evoked_plot_ave_fig_id = os.path.join(merged_evoked_folder_fig, condition_name +  '_' + 'merged_evoked_ave.pdf') 
    
    print("Saving figure @ " + str(merged_evoked_plot_fig_id))
    
    merged_evoked_fig = merged_evoked.plot_topomap(times='peaks', extrapolate='head', time_unit='ms', ch_type=ch_type, outlines='head', sphere=(0., 0., -0.00332, 0.18))
    plt.savefig(merged_evoked_plot_fig_id)
    plt.close()

    return merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id

def merge_evoked_component(evoked_list, cond, k, target_folder, ch_type='mag'):
    merged_evoked = mne.combine_evoked(evoked_list, weights= 'nave')
    condition_name = cu.get_condition_name(cond)
    merged_evoked_folder_fif = os.path.join(target_folder, 'fif')
    fs.ensure_dir(merged_evoked_folder_fif)

    merged_evoked_folder_fig = os.path.join(target_folder, 'fig')
    fs.ensure_dir(merged_evoked_folder_fig)
    
    comp_name = 'merged_comp_' + str(k+1) + '_' + condition_name
    file_name = 'merged_comp_' + str(k+1) + '_' + condition_name + '.fif'
    merged_evoked_file_id = os.path.join(merged_evoked_folder_fif, file_name)
    print("merged_evoked shape= " + str(merged_evoked.data.shape))
    print("Writing Merged Evoked for Condition =" + str(condition_name) + "; @ " + str(merged_evoked_file_id))
    mne.write_evokeds(merged_evoked_file_id, merged_evoked)
    
    file_np_name = 'merged_comp_' + str(k+1) + '_' + condition_name + '.npy'
    merged_evoked_np_id = os.path.join(merged_evoked_folder_fif, file_np_name)
    np.save(merged_evoked_np_id, merged_evoked.data)
    
    print("Merged Evoked Channels: " + str(merged_evoked.info['ch_names']))
    merged_evoked_plot_fig_id = os.path.join(merged_evoked_folder_fig, comp_name  + '.pdf')  
    print("Saving figure @ " + str(merged_evoked_plot_fig_id))
    
    merged_evoked_fig = merged_evoked.plot_topomap(times='peaks', extrapolate='head', time_unit='ms', ch_type=ch_type, outlines='head', sphere=(0., 0., -0.00332, 0.18))
    plt.savefig(merged_evoked_plot_fig_id)
    plt.close()
    
    return merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id