import os
import configparser
from os import path

cond_name = {'6':'aud',
            '4':'vis',
            '2':'aud_vis'
            }

SFREQ = 1000

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config



