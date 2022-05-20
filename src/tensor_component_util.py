import numpy as np
import os as os
import configparser
from os import path
import pandas as pd
import file_service as fs
import io_functions as io
import mne
import config_util as cu
import matplotlib.pyplot as plt

import plot_evoked_utils as ev
import factor as factor
from mne.channels import find_ch_adjacency


def create_3D_evoked_arr(lst):
    
    subject_list = []
    cnt = 0
    for e in lst:
        subject_evoked_data = e.data
        subject_list.append(subject_evoked_data)
        cnt = cnt + 1
        
    subject_data = np.dstack(subject_list)
    print("Subject 3D Data shape = " + str(subject_data.shape))
    
    return subject_data

def create_adjacency_grad(evoked):
    channels_idx = mne.pick_channels(evoked.info['ch_names'], include=[], exclude=[])
    selected_info = mne.pick_info(evoked.info, channels_idx)
    sensor_adjacency, ch_names = find_ch_adjacency(selected_info, ch_type='grad')
    use_idx = [ch_names.index(ch_name.replace(' ', ''))
           for ch_name in evoked.info.ch_names]
    sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
    return sensor_adjacency

def get_component_num_by_path(target_path, comp_num, factor_num):
    component_path = os.path.join(target_path, 'meg_' + 'fact_cpd_' + str(factor_num) + '.npy')
    component = np.load(component_path)
    component_data = component[:, comp_num - 1]
    return component_data

def get_component_path(target_path, factor_num):
    component_path = os.path.join(target_path, 'meg_' + 'fact_cpd_' + str(factor_num) + '.npy')
    return component_path


def get_component_by_path(target_path, factor_num):
    component_path = os.path.join(target_path, 'meg_' + 'fact_cpd_' + str(factor_num) + '.npy')
    component = np.load(component_path)
    return component

def get_component(target_path, factor_num):
    component_path = os.path.join(target_path, 'fact_' + str(factor_num) + '.npy')
    component = np.load(component_path, allow_pickle = True)
    return component

def get_reconstructed_component(target_path):
    component_path = os.path.join(target_path, 'x_hat.npy')
    component = np.load(component_path, allow_pickle = True)
    return component

def reconstruct_evoked(x, data, cond):
    subject_info = x.info
    
    print("Subject Info = " + str(subject_info))
    evoked_data = mne.EvokedArray(data, subject_info, x.tmin, kind='average', comment=cond,
                               nave=1, verbose=True)
    print(evoked_data)    
    return evoked_data


def generate_evoked(evoked_ref_path, data, cond):
    org_evoked = io.read_evokeds_by_path_and_channel_type_singles(evoked_ref_path, baseline=(None, 0), verbose=True)
    evoked = reconstruct_evoked(org_evoked, data, cond)
    return evoked

def generate_evoked_from_list(evoked_ref_path, data, cond):
    org_evoked = io.read_evokeds_by_path_and_channel_type_singles(evoked_ref_path, baseline=(None, 0), verbose=True)
    evoked = reconstruct_evoked(org_evoked[0], data, cond)
    return evoked


def reference_evoked(evoked_ref_path):
    evoked = io.read_evokeds_by_path_and_channel_type_singles(evoked_ref_path, baseline=(None, 0), verbose=True)
    return evoked[0]


def get_evoked(evoked_path):
    evoked = io.read_evokeds_by_path_and_channel_type_singles(evoked_path, baseline=(None, 0), verbose=True)
    return evoked[0]

def get_evoked_grad(evoked_path, cond):
    evoked = io.read_evokeds_by_path_and_channel_type_grad(evoked_path, cond, baseline=(None, 0), verbose=True)
    return evoked

   
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
    
    try:
        merged_evoked_fig = merged_evoked.plot_topomap(times='peaks', extrapolate='head', time_unit='ms', ch_type=ch_type, outlines='head', sphere=(0., 0., -0.00332, 0.18))
        plt.savefig(merged_evoked_plot_fig_id)
        plt.close()
    except:
        print("Error plotting evoked")
    
    return merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id

def create_merged_evoked(ref_evoked, data, target_folder, k, cond, cond_name, components_folder, condition_folder):
    subject_length  = data.shape[0]
    rows  = []
    
    csv_folder = os.path.join(condition_folder, 'csv')
    fs.ensure_dir(csv_folder)
    subject_df_path = os.path.join(csv_folder, 'subject_sources.csv')
    subject_df = pd.read_csv(subject_df_path)
    
    subject_source_dir_fif = os.path.join(target_folder, 'sources', 'fif')
    subject_source_dir_fig = os.path.join(target_folder, 'sources', 'fig')
    subject_source_dir_np = os.path.join(target_folder, 'sources', 'np')
    subject_source_dir_csv = os.path.join(target_folder, 'sources', 'csv')
    fs.ensure_dir(subject_source_dir_fif)
    fs.ensure_dir(subject_source_dir_np)
    fs.ensure_dir(subject_source_dir_csv)
    fs.ensure_dir(subject_source_dir_fig)
    
    source_column_fif = 'source_fif'
    source_column_np = 'source_np'
    subject_source_list_fif = []
    subject_source_list_np = []
    
    for i in range(subject_length):
        subject = data[i, :, :]
        print("subject shape" + str(subject))
        
        subject_fif_path = os.path.join(subject_source_dir_fif, 's' + '_' + str(i+1) + '.fif')
        subject_np_path = os.path.join(subject_source_dir_np, 's' + '_' + str(i+1) + '.npy')
        
        fig_id = os.path.join(subject_source_dir_fig, 's' + '_' + str(i+1) + '.pdf')
        evoked = reconstruct_evoked(ref_evoked, subject, cond)
        try:
            ev.plot_evoked_with_figure_id(evoked, target_folder, k, cond_name, fig_id, tmax=0.8)
        except:
            print("Error occurred while plotting for Subject: " + str(i))
        
        
        rows.append(evoked)
        evoked.save(subject_fif_path)
        print("Saved Subject fif " + str(i+1)  + " @ " + subject_fif_path)
        
        np.save(subject_np_path, evoked.data)
        print("Saved Subject np " + str(i+1)  + " @ " + subject_np_path)
        
        
        subject_source_list_fif.append(subject_fif_path)
        subject_source_list_np.append(subject_np_path)

            
    merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id = merge_evoked_component(rows, cond, k, target_folder, ch_type='grad')
    print("Generated subject sources. Done")
    
    component_csv_folder = os.path.join(components_folder, 'csv')
    fs.ensure_dir(component_csv_folder)
    subject_df_path_comp = os.path.join(component_csv_folder, 'subject_sources.csv')
    subject_df_path_comp_num = os.path.join(subject_source_dir_csv, 'subject_sources.csv')
    
    subject_df[source_column_fif] = subject_source_list_fif
    subject_df[source_column_np] = subject_source_list_np
    subject_df.to_csv(subject_df_path_comp, index=False)
    print("Saved Subject Metadata" +  " @ " + subject_df_path_comp)
    
    subject_df.to_csv(subject_df_path_comp_num, index=False)
    print("Saved Subject Metadata " +  " @ Component Folder " + subject_df_path_comp)
    
    subject_tensor_folder =  os.path.join(target_folder, 'fif')
    fs.ensure_dir(subject_tensor_folder)
    subject_tensor_file_id = os.path.join(subject_tensor_folder, 'subject_tensor.npy')
    np.save(subject_tensor_file_id, data)
    
    return merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id



def reconstruct_component(evoked_ref_path, x_a, x_b, x_c, target_folder, k, cond, temporal_comp, components_folder, condition_folder):
    ref_evoked = reference_evoked(evoked_ref_path)
    print("Reference Evoked path: " + str(evoked_ref_path))
    
    subject_info = ref_evoked.info
    cond_name = cu.get_condition_name(cond)
    a = x_a
    b = x_b
    c = x_c
    a_k = a[:,k]
    b_k = b[:,k]
    c_k= c[:,k]
    aD = a_k[:, np.newaxis]
    bD = b_k[:, np.newaxis]
    cD = c_k[:, np.newaxis]
    x = (bD*cD.T).T
    print("x shape" + str(x.shape))
    x_est = x[:, np.newaxis]
    subjects = np.dot(aD, x_est.T)
    subject_group_data = np.transpose(subjects, [0, 2, 1])
    print("subjects.shape" + str(subjects.shape))
    print("subjects group data.shape" + str(subject_group_data.shape))
    subject_tp_evoked = reconstruct_evoked(ref_evoked, x, cond)
    comp_name = 'comp_' + str(k+1) + '_' + cond_name + '.fif'
    fif_folder = os.path.join(target_folder, 'fif')
    fs.ensure_dir(fif_folder)
    subject_path = os.path.join(fif_folder, comp_name)
    subject_tp_evoked.save(subject_path)
    merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id = create_merged_evoked(ref_evoked, subject_group_data, target_folder, k, cond, cond_name, components_folder, condition_folder)
    ev.plot_evoked_original(ref_evoked, target_folder, k, cond_name, tmax = 0.8, npeaks = 3)
    ev.plot_evoked(merged_evoked, target_folder, k, cond_name, tmax=0.8)
    ev.plot_evoked_with_topo(merged_evoked, target_folder, k, cond_name, tmax=0.8)
    temporal_evoked = generate_evoked_from_list(evoked_ref_path, temporal_comp.T, cond_name)
    c_t = c.T
    
    evoked = {cond_name.upper():merged_evoked}
    extracted_evoked = evoked[cond_name.upper()].crop(0,1)
        
    sp_comp = c_t[k, :]
    sp  = sp_comp[:, np.newaxis]
    tc1 = temporal_evoked.data[k,:]
    tc  = tc1[:, np.newaxis]
    print("Spatial Component shape = " + str(sp.shape))
    print("Temporal Component shape = " + str(tc.shape))
    comp_name = 'col_' + str(k)
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = os.path.join(fig_folder, comp_name)
    tp = generate_evoked_from_list(evoked_ref_path, sp, cond_name)
    
    fact_sol = factor.Factor(extracted_evoked, temporal_comp.shape[1], k, info = subject_info)
    
    cnt = 0
    for t in temporal_evoked.data:
        if cnt == k:
            sp_comp = c_t[cnt-1, :]
            sp  = sp_comp[:, np.newaxis]
            fig_id = os.path.join(fig_folder, comp_name)
            temporal_path = os.path.join(fif_folder, comp_name + '.npy')
            np.save(temporal_path, t)
            fact_sol.plot_factor(cond_name.upper(), cnt, extracted_evoked,  t, tp, fig_id = fig_id,title=None,
                    ch_type='grad', p_value=None)   
        
        cnt = cnt + 1
        
    
    return merged_evoked

def read_evoked_list(evoked_list):
    
    subject_list = []
    for s in evoked_list:
        subject_evoked = get_evoked(s)
        subject_list.append(subject_evoked)
    
    return subject_list


def read_evoked_list_grad(evoked_list, cond):
    
    subject_list = []
    for s in evoked_list:
        subject_evoked = get_evoked_grad(s, cond)
        subject_list.append(subject_evoked)
    
    return subject_list

def save_map(x, data, cond, name, thr, target_folder):
    
    file_id = os.path.join(target_folder, name + '_' + str(thr) + '.fif')
    map_ev = reconstruct_evoked(x, data, cond)
    map_ev.save(file_id)
    print("Saved Evoked " + name + "@" + file_id)
    return map_ev
           
def find_peaks(evoked, npeaks, tmax):
    """Find peaks from evoked data.
    Returns ``npeaks`` biggest peaks as a list of time points.
    """
    evoked_bk = evoked.copy()
    evoked_bk = evoked_bk.crop(0,tmax)
    from scipy.signal import argrelmax
    gfp = evoked_bk.data.std(axis=0)
    order = len(evoked_bk.times) // 30
    if order < 1:
        order = 1
    peaks = argrelmax(gfp, order=order, axis=0)[0]
    if len(peaks) > npeaks:
        max_indices = np.argsort(gfp[peaks])[-npeaks:]
        peaks = np.sort(peaks[max_indices])
    times = evoked_bk.times[peaks]
    if len(times) == 0:
        times = [evoked_bk.times[gfp.argmax()]]
    print(times)
    return times

def create_3D_evoked(lst):
    
    subject_list = []
    cnt = 0
    for e in lst:
        subject_evoked_data = e.data
        subject_list.append(subject_evoked_data)
        cnt = cnt + 1
    
    subject_data = np.dstack(subject_list)
    subject_group_data = np.transpose(subject_data, [2, 1, 0])
    N = subject_group_data.shape[2]
    C = subject_group_data.shape[1]
    T = subject_group_data.shape[0]
        
    print(" Subject Data shape = " + str(subject_data.shape))

    return subject_group_data

    