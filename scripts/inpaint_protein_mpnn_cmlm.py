import os
import sys
import pickle

import numpy as np
import pytorch_lightning as pl

from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

# function
def string2list(editable_string):
    """convert editable Aa string (left-colse, right-open interval) to editable id list (two-side closed).

    Args:
        editable_string: 1-based coordinates

    Returns:
        tuple of start ids and end ids (1-based coordinates)
        
    Examples:
        1-2,4-7,15-21 => ([1, 4, 15], [1, 6, 20])
    """
    interval_ls = [s.split('-') for s in editable_string.split(',')]
    start_ids = [int(interval[0]) for interval in interval_ls]
    end_ids = [int(interval[1])-1 for interval in interval_ls]
    return (start_ids, end_ids)


# model checkpoint path
exp_path = sys.argv[1]
# pdb file list path (col3 is editable residue id string, left-colse & right-open interval)
pdb_file_list = sys.argv[2]
# outfile
outfile = sys.argv[3]
os.makedirs(os.path.dirname(outfile), exist_ok=True)


CANONICAL_AA_SET = set('ACDEFGHIKLMNPQRSTVWY')

# 1. instantialize designer
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=1,
        strategy='mask_predict', 
        temperature=0,
        eval_sc=False,  
    )
)
designer = Designer(experiment_path=exp_path, cfg=cfg)

# infer
pred_ls = []
with open(pdb_file_list, 'r') as PDB_IN:
    for line in PDB_IN.readlines():
        print('------------- ' + line.strip().replace("\t", ";") + '-------------')
        pdb_path, edit_chain_ids_str, edit_res_ids_str, temp_str, n_samp_str, seed_str = line.strip().split('\t')
        pl.seed_everything(seed=int(seed_str))
        
        # 2. load structure from pdb file
        designer.set_structure(pdb_path)
        
        # 3. generate sequence from given structure
        start_id_ls, end_id_ls = string2list(edit_res_ids_str)
        seq_ls = []
        n_invalid = 0
        for i in range(int(n_samp_str)):
            # inpaint require start/end ids in 0-based coordinates
            out, ori_seg, designed_seg = designer.inpaint(
                start_ids=[i-1 for i in start_id_ls], 
                end_ids=[i-1 for i in end_id_ls], 
                generator_args={'temperature': float(temp_str)}
            )
            if (len(out[0][0]) != len(designer._structure['seq'])) or (len(set(out[0][0]) - CANONICAL_AA_SET) != 0):
                n_invalid += 1
                continue
            seq_ls.append([f'seq_{i}', out[0][0]])
        pred_ls.append(
            {
                'seq_info': seq_ls,
                'start_ids': start_id_ls,
                'end_ids': end_id_ls,
                'pdb_path': pdb_path,
                'temperature': float(temp_str),
                'seed': int(seed_str)
            }
        )
        print(f'{n_invalid:_} ({n_invalid/int(n_samp_str):.1%}) dropped due to special tokens or non-cononical AA')

with open(outfile, 'wb') as PKL_OUT:
    pickle.dump(pred_ls, PKL_OUT, protocol=pickle.HIGHEST_PROTOCOL)
