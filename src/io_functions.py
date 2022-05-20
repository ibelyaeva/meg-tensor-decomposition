from __future__ import print_function

import mne
from os.path import join
import pickle

    
def read_evokeds_by_path(file_path,condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    return evokeds   

def read_evokeds_by_path_channels(file_path, channels, type='mag', condition=None, baseline=None, kind='average', 
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    evokeds.pick_types(type, verbose=verbose)
    
    print("Evoked Channels = " + str(channels))
    evokeds.pick_channels(channels)
    
    return evokeds   

def read_evokeds_by_path_and_type(file_path, type='mag', condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    evokeds.pick_types(type, verbose=verbose)
    
    return evokeds 

def read_evokeds_by_path_and_channel_type(file_path,type='mag', condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    evokeds.pick_types(type, verbose=verbose)
    
    return evokeds 

def read_evokeds_by_path_and_channel_type_singles(file_path, condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    return evokeds 

def read_evokeds_by_path_and_channel_type_grad(file_path, condition=None, baseline=None, kind='average',
                 proj=True, allow_maxshield=False, verbose=None):
    
    evokeds = mne.read_evokeds(file_path, condition, baseline, kind,
                 proj=proj, allow_maxshield=allow_maxshield, verbose=verbose)
    
    print(evokeds)
    evokeds.pick_types('grad', verbose=verbose)
    
    return evokeds