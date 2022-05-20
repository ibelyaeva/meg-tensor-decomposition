import tensortools as tt
import numpy as np
import pandas as pd
import os


def compute_metrics(data, target_folder, rmin=1, rmax = 11, replicates=10):
    decomp = tt.Ensemble(fit_method="cp_als")
    decomp.fit(data, ranks=range(rmin, rmax), replicates=replicates)
    collect_similarity(decomp, target_folder)
    collect_objectives(decomp, target_folder)
    
    
def collect_similarity(t, target_folder):
    x, sim, mean_sim, std_sim = [], [], [], []
    
    rows = []
    row_dict = {}
    rank_dict =  {}
    cnt = 0
    for rank in sorted(t.results):
        s = t.similarities(rank)[1:]
        sim.extend(s)
        x.extend(np.full(len(s), rank))
        mean_sim.append(np.mean(s))
        std_sim.append(np.std(s))
        rank_dict[cnt] = rank
        cnt = cnt + 1
        
    cnt = 0
    row_rank_list = []
    for i in sim:
        row_dict['sim'] = i
        row_rank_list.append(x[cnt])
        rows.append(i)
        cnt = cnt + 1
        print("sim")
        print(row_dict)
        
    similarity_df = pd.DataFrame()  
    similarity_df['r'] = row_rank_list
    similarity_df['sim'] = rows
    similarity_df_path = os.path.join(target_folder, 'similarity_cost.csv')
    similarity_df.to_csv(similarity_df_path, index = False)
    
    rows = []
    mean_sim_list = []
    std_sim_list = []
    cnt = 0
    for i in mean_sim:
        mean_sim_list.append(i)
        std_sim_list.append(std_sim[cnt])
        cnt = cnt + 1
       
        
    summary_df = pd.DataFrame() 
    summary_df['mean_sim'] = mean_sim_list
    summary_df['std_sim'] = std_sim_list
    summary_df_path = os.path.join(target_folder, 'similarity_summary_cost.csv')
    summary_df.to_csv(summary_df_path)
    
        
    return similarity_df, summary_df
    
def collect_objectives(t, target_folder):
    x, obj, min_obj = [], [], []
    
    for rank in sorted(t.results):
        o = t.objectives(rank)
        obj.extend(o)
        x.extend(np.full(len(o), rank))
        min_obj.append(min(o))
        
    cnt = 0
    rse_list = []
    rank_list = []
    
    for i in obj:
        rank_list.append(x[cnt])
        rse_list.append(1.0 - i)
        cnt = cnt + 1
        
    summary_df = pd.DataFrame()  
    summary_df['r'] = rank_list
    summary_df['rse'] = rse_list
    
    summary_df_path = os.path.join(target_folder, 'objective_cost.csv')
    summary_df.to_csv(summary_df_path)
    return summary_df