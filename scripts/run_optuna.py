# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
run_optuna.py
--------------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import optuna

root_dir = Path(__file__).resolve().parent.parent
log_dir = os.path.join(root_dir,'logs')

sys.path.insert(0,str(root_dir))

from src import load_datasets
from src import dataset
from src import averagers
from src import train

REPLICATES_PER_TRIAL = 5

def train_LAvD(
        train_data:dataset.dms_dataset,
        val_data:dataset.dms_dataset,
        averager_output_dim:int,
        dense_out:int,
        dense_lyr:int,
        drp:float,
        use_token_pll:bool,
        use_total_pll:bool,
        starting_lr:float,
        epochs:int,
        use_ramcache:bool=True,
        )->averagers.LearnableAveragerDense:
    """
    Trains a learanable averager
    """

    model = averagers.LearnableAveragerDense(
        encoder_name ='esm2_t33_650m_ur50d',
        averager_output_dim=averager_output_dim,
        out=dense_out,
        lyr=dense_lyr,
        drp=drp,
        use_total_pll=use_total_pll,
        use_token_pll=use_token_pll
    )

    if torch.cuda.is_available():
        model.to('cuda')
    model.use_ramcache(use_ramcache)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=starting_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        )
    
    return train.train_model(
        model,
        training_data=train_data,
        val_data=val_data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        total_epochs=epochs,
        train_batch_size=256,
        val_batch_size=256)


def optuna_wrapper_c1r2(trial,log_file:str,token_pll:bool,total_pll:bool)->float:
    print("Trial {} ".format(trial.number))

    #Set up hyperparameters
    averager_output_dim = trial.suggest_int('averager_output_dim',100,1280)
    dense_out = trial.suggest_int('dense_out',100,480)
    dense_lyr = trial.suggest_int('dense_lyr',2,6)
    drp = trial.suggest_float('drp','0','0.5')
    start_lr = trial.suggest_float('lr',1e-6,1e-3,log=True)
    epochs = trial.suggest_int('epochs',50,250)

    corrs = list()
    for replicate in range(REPLICATES_PER_TRIAL):
        c1 = load_datasets.load_c1()
        r2 = load_datasets.load_r2()
        r10 = load_datasets.load_r10()
        c1.shuffle(replicate)
        r2.shuffle(replicate)
        r10.shuffle(replicate)
        
        val_data,_ = r10.split_at_n(1977)#Hold out 1,977 randoms from 2-10 as validation

        r2,_ = r2.split_at_n(1756)
        train_data = c1.copy()
        train_data.extend(r2)
        trained_model = train_LAvD(
                train_data=train_data,
                val_data=val_data,
                averager_output_dim=averager_output_dim,
                dense_out=dense_out,
                dense_lyr=dense_lyr,
                drp=drp,
                use_token_pll=token_pll,
                use_total_pll=total_pll,
                starting_lr=start_lr,
                epochs = epochs,
        )
        corr = train.get_corr(trained_model,val_data)
        corrs.append(corr)

    write_log(log_file,trial,np.mean(corrs))

    return np.mean(corrs)

def write_log(log_file,trial,corr):
    # Log rial information to a file
    log_path = os.path.join(log_dir,log_file)
    if not os.path.exists(log_path):
        with open(log_path,'a') as f:
            f.write('trial_number,trial_result,')
            for param in trial.params:
                f.write(str(param)+',')
            f.write('\n')
    with open(log_path, "a") as f:
        f.write(str(trial.number)+','+str(corr)+',')
        for param in trial.params:
            f.write(str(trial.params[param])+',')
        f.write('\n')



def main():
    if not os.path.exists(log_dir):os.mkdir(log_dir)

    study_name = 'aav2_c1r2_lav'
    storage = 'sqlite:////{}'.format(
        os.path.join(log_dir,'{}.db'.format(study_name))
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler = optuna.samplers.TPESampler(),
        direction='maximize'
    )
    study.optimize(
        lambda trial:optuna_wrapper_c1r2(
            trial,
            log_file='{}.csv'.format(study_name),
            token_pll=False,
            total_pll=False,
            ),
        n_trials=50
    )
    

if __name__=="__main__":
    main()
    







