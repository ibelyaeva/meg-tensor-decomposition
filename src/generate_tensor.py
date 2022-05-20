import configparser
from os import path
import config_util as cu
import numpy as np
import convert_to_mat
import os
import io_functions as io
import pandas as pd
import file_service as fs
import copy
import scipy.stats as stats
import plot_tensor_results as pt

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def read_data(target_path, target_dir, cond, type_):
    cond_path = os.path.join(target_dir, 'data')
    print("Condition Path: " + cond_path)
    fs.ensure_dir(cond_path)
    csv_dir = os.path.join(target_dir, 'csv')
    fs.ensure_dir(csv_dir)
    subject_meta_path = os.path.join(csv_dir, 'subject_sources.csv')
    
    session_df = pd.read_csv(target_path)
    session_df.to_csv(subject_meta_path, index = False) # save original subject list
    
    subject_list = session_df.values.tolist()
    
    cnt = 0
    labels = []
    subjects = []
    subject_evoked_list = []
    for s in subject_list:
        subject_name = s[3]
        subject_path = s[4]
        print("Subject Name: " + str(subject_name) + "; " + "Subject Path: " + subject_path)
        if os.path.isfile(subject_path):
            subject_evoked = io.read_evokeds_by_path_and_type(subject_path, type=type_,baseline = (None, 0), kind='average', 
                                                                          condition=cond, verbose=True)
            print(subject_evoked.info)
            subject_evoked_data = subject_evoked.data
            subjects.append(subject_evoked_data)
            subject_evoked_list.append(subject_evoked)
            labels.append(subject_name)
            cnt = cnt + 1
            print("Subject Count: " + str(cnt))
            
    
    subject_data = np.dstack(subjects)
    
    subject_group_data = np.transpose(subject_data, [2, 1, 0])
    N = subject_group_data.shape[2]
    C = subject_group_data.shape[1]
    T = subject_group_data.shape[0]
    
    print(" Subject Data shape = " + str(subject_data.shape))
    return subject_group_data, subject_evoked_list

def normalize(x, axis = 1):
    N = x.shape[0]
    C = x.shape[2]
    T = x.shape[1]
    x_temp = np.reshape(x, (N, C*T))
    res = stats.zscore(x_temp, axis = axis)
    result = np.reshape(res, (N, T, C))
    return result
    

def create_input(cond, meta_path, target_dir, type_):
    
    cond_path = os.path.join(target_dir, 'data')
    print("Condition Path: " + cond_path)
    fs.ensure_dir(cond_path)
    
    subject_data, subject_list = read_data(meta_path, target_dir, cond, type_)
    subject_data_z = normalize(subject_data)
    subject_data_np_path = os.path.join(cond_path, 'subject_data_' + str(type_) + '.npy')

    np.save(subject_data_np_path, subject_data_z)
    print("Saved Subject Data in Numpy @" + str(subject_data_np_path) + " Subject Data shape = " + str(subject_data.shape))
    
    merged_evoked, merged_evoked_folder_fif, merged_evoked_file_id = pt.merge_evoked(subject_list, cond, target_dir, ch_type='grad')
    return merged_evoked

def create_condition_data(data_type, target_dir, cond, type_):
    session_path = config.get('sessions',data_type)
    cond_name = cu.get_condition_name(cond)
    print("Generating Data for Condition: " + str(cond) + "; Cond Name: " + str(cond_name) + "; " + "Sensor Type: " + str(type_))
    condition_evoked = create_input(cond, session_path, target_dir, type_)
    return condition_evoked