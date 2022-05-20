import numpy as np
import os as os
import configparser
from os import path

def get_matlab_solution_path(target_dir, cond_name, comp_num):
    subject_group_name = cond_name + '_' + str(comp_num) + '_ica_br1' + '.mat' 
    subject_group_path = os.path.join(target_dir, subject_group_name)
    return subject_group_path