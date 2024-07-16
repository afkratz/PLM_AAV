# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
run_optuna.py
--------------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path
from typing import Callable,Tuple,Dict
from abc import ABC,abstractmethod
import pandas as pd
import torch
import numpy as np
import optuna
from torch.nn.modules import Module

from sklearn.metrics import roc_auc_score


root_dir = Path(__file__).resolve().parent.parent
log_dir = os.path.join(root_dir,'logs')

sys.path.insert(0,str(root_dir))

from src import load_datasets
from src import dataset
from src import averagers
from src import train

REPLICATES_PER_TRIAL = 5


class OptunaModel(ABC):
    """
    Abstract base class wrapping around a model with parameters to be optimized by Optuna
    """
    @abstractmethod
    def suggest_parameters(self,trial:optuna.Trial)->Dict:
        pass
    
    @abstractmethod
    def new_model(self,parameters:Dict)->torch.nn.Module:
        pass

    @abstractmethod
    def get_model_name(self)->str:
        pass

class OptunaLAV(OptunaModel):
    def __init__(self,use_total_pll:bool,use_token_pll:bool,model_name:str):
        self.use_total_pll=use_total_pll
        self.use_token_pll=use_token_pll
        self.model_name=model_name

    def suggest_parameters(self, trial: optuna.Trial) -> Dict:
        parameters = dict()
        parameters['averager_output_dim'] = trial.suggest_int('averager_output_dim',100,1280)
        parameters['dense_out'] = trial.suggest_int('dense_out',100,480)
        parameters['dense_lyr'] = trial.suggest_int('dense_lyr',2,6)
        parameters['drp'] = trial.suggest_float('drp',0,0.5)
        parameters['start_lr'] = trial.suggest_float('start_lr',1e-6,1e-3,log=True)
        parameters['epochs'] = trial.suggest_int('epochs',50,250)
        return parameters
    
    def new_model(self,parameters:Dict)->averagers.LearnableAveragerDense:
        model = averagers.LearnableAveragerDense(
            encoder_name ='esm2_t33_650m_ur50d',
            averager_output_dim=parameters['averager_output_dim'],
            out=parameters['dense_out'],
            lyr=parameters['dense_lyr'],
            drp=parameters['drp'],
            use_total_pll=self.use_total_pll,
            use_token_pll=self.use_token_pll
            )
        model.use_ramcache(True)
        return model

    def get_model_name(self) -> str:
        return self.model_name

class OptunaSAV(OptunaModel):
    def __init__(self,use_pll:bool,model_name):
        self.use_pll=use_pll
        self.model_name=model_name

    def suggest_parameters(self, trial: optuna.Trial) -> Dict:
        parameters = dict()
        parameters['dense_out'] = trial.suggest_int('dense_out',100,480)
        parameters['dense_lyr'] = trial.suggest_int('dense_lyr',2,6)
        parameters['drp'] = trial.suggest_float('drp',0,0.5)
        parameters['start_lr'] = trial.suggest_float('start_lr',1e-6,1e-3,log=True)
        parameters['epochs'] = trial.suggest_int('epochs',50,250)
        return parameters
    
    def new_model(self, parameters: Dict) -> averagers.SimpleAveragerDense:
        model = averagers.SimpleAveragerDense(
            encoder_name ='esm2_t33_650m_ur50d',
            out=parameters['dense_out'],
            lyr=parameters['dense_lyr'],
            drp=parameters['drp'],
            use_pll=self.use_pll
            )
        model.use_ramcache(True)
        return model

    def get_model_name(self) -> str:
        return self.model_name



def train_model(
        model:torch.nn.Module,
        train_data:dataset.dms_dataset,
        val_data:dataset.dms_dataset,
        starting_lr:float,
        epochs:int,
        )->torch.nn.Module:
    
    if torch.cuda.is_available():
        model.to('cuda')
    
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=starting_lr
        )
    
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
        val_batch_size=256,
        verbose=False
        )


def task_wrapper(
        optunamodel:OptunaModel,
        trial:optuna.Trial,
        load_dataset:Callable[[int],Tuple[dataset.dms_dataset,dataset.dms_dataset,str]]
        )->float:
    parameters = optunamodel.suggest_parameters(trial)
    corrs = list()
    for replicate in range(REPLICATES_PER_TRIAL):
        model = optunamodel.new_model(parameters) 
        training_data,val_data,dataset_name = load_dataset(replicate)
        model = train_model(model,train_data=training_data,
                            val_data=val_data,
                            starting_lr=parameters['start_lr'],
                            epochs=parameters['epochs'])
        corr = train.get_corr(model,eval_data=val_data,batch_size=128)
        corrs.append(corr)
    log_file = "optuna_log_{}_{}.csv".format(
        optunamodel.get_model_name(),
        dataset_name)
    write_log(log_file,trial,np.mean(corrs))
    return np.mean(corrs)


def load_c1r2(replicate:int)->Tuple[dataset.dms_dataset,dataset.dms_dataset,str]:
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
    return (train_data,val_data,'c1r2')

def load_c1r2_data()->Dict[str,dataset.dms_dataset]:
    return{
        'lr_c1r2':load_datasets.load_lr_c1r2(),
        'cnn_c1r2':load_datasets.load_cnn_c1r2(),
        'rnn_c1r2':load_datasets.load_rnn_c1r2(),
    }


def write_log(log_file,trial,corr):
    # Log trial information to a file
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
    
    model_list = [
        OptunaSAV(use_pll=False,model_name='SAV_NoPLL'),
        OptunaSAV(use_pll=True,model_name='SAV_PLL'),
        OptunaLAV(use_token_pll=False,use_total_pll=False,model_name='LAV_NoPLL'),
        OptunaLAV(use_token_pll=False,use_total_pll=True, model_name='LAV_TotalPLL'),
        OptunaLAV(use_token_pll=True, use_total_pll=False,model_name='LAV_TokenPLL'),
    ]

    dataset_list = [
        load_c1r2,
    ]#Could expand this later

    results = pd.DataFrame()
    for model in model_list:
        for dataset_function in dataset_list:
            _,_,dataset_name = dataset_function(0)

            study_name = 'aav2_{}_{}'.format(
                model.get_model_name(),
                dataset_name
                )
            
            storage = 'sqlite:////{}'.format(
            os.path.join(log_dir,'{}.db'.format(study_name))
            )

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler = optuna.samplers.TPESampler(seed=13),
                direction='maximize',
            )
            
            study.optimize(
                lambda trial: task_wrapper(
                    optunamodel=model,
                    trial=trial,
                    load_dataset=dataset_function
                ),
                n_trials=50
            )

            params = study.best_params
            test_data = load_c1r2_data()

            for fraction in np.linspace(0.2,1,5):
                for k in test_data:
                    results.at["{}_{}_{}".format(model.get_model_name(),k,str(fraction)[:3]),"model_name"]=model.get_model_name()
                    results.at["{}_{}_{}".format(model.get_model_name(),k,str(fraction)[:3]),"test_set"]=k
                    results.at["{}_{}_{}".format(model.get_model_name(),k,str(fraction)[:3]),"fraction"]=fraction
                
                for replicate in range(REPLICATES_PER_TRIAL):
                    final_model = model.new_model(params)
                    train_data,val_data,dataset_name = dataset_function(-replicate)
                    train_data = train_data.take_fraction(fraction)
                    final_model = train_model(final_model,train_data,val_data,params['start_lr'],params['epochs'])
                    final_model.use_ramcache(False)
                    
                    for k in test_data:
                        print("{}_{}_{}, {}".format(model.get_model_name(),k,str(fraction)[:3],replicate))
                        data_set = test_data[k]
                        x = train.get_model_pred(final_model,eval_data=data_set,batch_size=128).cpu()
                        y = np.array(data_set.is_viable).astype(int)
                        auc = roc_auc_score(y,x)
                        results.at["{}_{}_{}".format(model.get_model_name(),k,str(fraction)[:3]),replicate]=auc
                
                results.to_csv('optuna_results.csv')


if __name__=="__main__":
    main()
    







