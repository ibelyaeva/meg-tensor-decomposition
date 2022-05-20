import numpy as np
import tensorly as tl
import configparser
from os import path
from tensorly.decomposition import CP, parafac
from tensorly.random import random_cp
from tensorly import cp_to_tensor
import pandas as pd
import os
import file_service as fs
import generate_tensor as gt
import config_util as ct
import cost_compute_util as cct
import backrecontruct_dataset as bk
import compute_metrics as cm
import scipy.stats as stats
import config_util as cu

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config


def decompose_task(cond, rank, runs, input_type):
    
    analysis_folder = config.get('cpd', 'analysis_folder')
    condition_name = ct.get_condition_name(cond)
  
    print("Started CPD decomposition for " + condition_name + "Tensor Rank"  + str(rank))
    #Create directories in advance
    comp_folder = os.path.join(analysis_folder, condition_name, 'data', 'components', 'r')
    fs.ensure_dir(comp_folder)
    target_dir = "tensor_rank/" + str(rank)
    batch_dir = fs.create_batch_directory(comp_folder,target_dir)
    print("Batch Directory", batch_dir)
    batch_dir_by_run = os.path.join(batch_dir, "run")
    run_folder = batch_dir_by_run + "/" + str(runs)
    fs.ensure_dir(run_folder)
    condition_evoked = gt.create_condition_data(input_type, batch_dir_by_run, cond , 'grad')
    cond_path = os.path.join(analysis_folder, condition_name)
    compute_solution(cond, run_folder, batch_dir_by_run, rank)
    
   
def normalize(x, axis = 1):
    N = x.shape[0]
    C = x.shape[2]
    T = x.shape[1]
    x_temp = np.reshape(x, (N, C*T))
    res = stats.zscore(x_temp, axis = axis)
    result = np.reshape(res, (N, T, C))
    return result 

def compute_solution(cond, target_folder, parent_folder,rank):
    condition_name = cu.get_condition_name(cond)
    run_csv_folder = os.path.join(target_folder, 'csv')
    fs.ensure_dir(run_csv_folder)
    x_true_path = os.path.join(parent_folder, 'data', "subject_data_grad.npy")
    x_true = np.load(x_true_path)
    print("Dataset shape = " + str(x_true.shape))
    
    x_tensor_true = tl.tensor(x_true)
    
    print("Computing CPD solution wit rank = " + str(rank) + "; Condition: " + condition_name)
    
    cp_decomp = CP(rank=rank, n_iter_max=1000, tol=1.0e-8, linesearch=True)
    x_hat = cp_decomp.fit_transform(x_tensor_true)
    
    print("X-hat shape" + str(x_hat.shape))
    print("Computing CPD solution with rank = " + str(rank) + " Done.")
    
    save_results(target_folder, x_hat, x_true)
    
    bk.reconstruct_component_by_condition(cond, target_folder, parent_folder)
    
    print("Reconstructed CPD solution wit rank = " + str(rank) + " Done.")

def save_results(target_comp_dir, x_hat, x_true, rmax = 3):
    
    x_pred = cp_to_tensor(x_hat)
    
    A = x_hat[1][0]
    B = x_hat[1][1]
    C = x_hat[1][2]
   
    x1_path = os.path.join(target_comp_dir, "fact_1.npy")
    x2_path = os.path.join(target_comp_dir, "fact_2.npy")
    x3_path = os.path.join(target_comp_dir, "fact_3.npy")
    
    
    np.save(x1_path, A)
    np.save(x2_path, B)
    np.save(x3_path, C)
    
    
    file_path = os.path.join(target_comp_dir, "x_hat.npy") 
    np.save(file_path, x_pred)
  
    #save normalized original object
    file_path = os.path.join(target_comp_dir, "x_true.npy") 
    np.save(file_path, x_true)
    
    file_path = os.path.join(target_comp_dir, "x_true_org.npy") 
    np.save(file_path, x_true)
    
    csv_folder = os.path.join(target_comp_dir, 'csv', 'cost')
    fs.ensure_dir(csv_folder)
    
    rse_cost_x = cct.relative_error(x_pred, x_true)
    
    rows = []
    row_dict = {}
    row_dict['rse_cost_x'] = rse_cost_x
           
    rows.append(row_dict)
    cost_df = pd.DataFrame(rows)
    
    cost_df_path = os.path.join(csv_folder, 'rse_cost.csv')
    cost_df.to_csv(cost_df_path)
    
    print("Saved CPD Solution @" + target_comp_dir)
    
    print("Computing Similarity Cost @" + csv_folder)
    cm.compute_metrics(x_true, csv_folder, rmin=1, rmax = rmax, replicates=3)
    
    print("Computed Similarity Cost @" + csv_folder)
    return csv_folder  
  
def get_solution(target_comp_dir):
    
    A_path = os.path.join(target_comp_dir, "fact_1.npy")
    B_path = os.path.join(target_comp_dir, "fact_2.npy")
    C_path = os.path.join(target_comp_dir, "fact_3.npy")
    
    A = np.load(A_path)
    B = np.load(B_path)
    C = np.load(C_path)
    
    x_hat_path = os.path.join(target_comp_dir, "x_hat.npy")
    x_true_path = os.path.join(target_comp_dir, "x_true.npy")
    x_org_path = os.path.join(target_comp_dir, "x_true_org.npy")
    
    x_hat = np.load(x_hat_path)
    x_true = np.load(x_true_path)
    x_org = np.load(x_org_path)
    
    return A, B, C, x_hat, x_true, x_org


if __name__ == '__main__':
    root_path = config.get('tensor-analysis', 'root_dir')
 
#decompose vis condition
decompose_task('4', 2, 1,  't1_cpd')

#decompose aud condition
decompose_task('6', 2, 1, 't1_cpd')

#decompose av condition
decompose_task('2', 2, 1, 't1_cpd')
