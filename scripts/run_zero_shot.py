# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
run_zero_shot.py
--------------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root_dir))

from src import zero_shot
from src import esm_enc



data_set = pd.read_csv(os.path.join(root_dir,'data','single_processed.csv'))

seqs = data_set['full_sequence'].to_list()
for encoder in esm_enc.encoder_names:
    zsh = zero_shot.PLLZeroShot(encoder_name=encoder)
    zsh.to('cuda')
    x = zsh.predict(seqs).cpu()
    y=np.array(data_set['is_viable']).astype(int)
    print(encoder,roc_auc_score(y,x))