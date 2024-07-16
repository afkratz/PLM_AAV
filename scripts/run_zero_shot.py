# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
run_zero_shot.py
--------------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root_dir))

from src import zero_shot

odf = pd.DataFrame()
encoder_name = 'esm2_t33_650m_ur50d'
zsh = zero_shot.PLLZeroShot(encoder_name=encoder_name)
zsh.to('cuda')
for category in ['single',
                 'random_doubles',
                 'random_up_to_ten',
                 'lr_c1r2',
                 'rnn_c1r2',
                 'cnn_c1r2']:
    data_set = pd.read_csv(os.path.join(root_dir,'data','{}_processed.csv'.format(category)))

    seqs = data_set['full_sequence'].to_list()
    x = zsh.predict(seqs).cpu()
    y=np.array(data_set['is_viable']).astype(int)
    auc = roc_auc_score(y,x)
    odf.at[category,encoder_name]=auc
    print(encoder_name,auc)
    odf.to_csv('rzs_output.csv')
