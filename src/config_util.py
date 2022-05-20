import os
import configparser
from os import path
import io_functions as io

cond_name = {'6':'aud',
            '4':'vis',
            '2':'aud_vis',
            '9':'sum',
            '10':'add'
            }

SFREQ = 1000

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config


def get_config():
    config_loc = path.join('config')
    config_filename = 'solution.config'
    config_file = os.path.join(config_loc, config_filename)
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config
    return config

def get_sample_subject_path():
    subject_path = config.get('channels', 'sample_subject_path')
    return subject_path

def get_channels(cond, type_):
    subject_path = get_sample_subject_path()
    print("Subject Path = " + str(subject_path))
    subject = io.read_evokeds_by_path_and_channel_type(subject_path, type=type_,baseline = (None, 0), kind='average', 
                                                                           condition=cond, verbose=True) 
    
    channels = subject.info['ch_names']
    return channels

def get_condition_name(name):
    return cond_name[name]

def get_channels_by_index(cond, type_, channels_list):
    all_channels = get_channels(cond, type_)
    
    selected_channels = []
    for c in channels_list:
        selected_channels.append(all_channels[c])
        
    return selected_channels