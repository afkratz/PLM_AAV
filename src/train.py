# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
train_utils.py
--------------------------------------------------------------------------------
"""


import torch

from torch.utils.data import DataLoader
from progress.bar import Bar
from . import dataset

from typing import Tuple

def train_model(
        model:torch.nn.Module,
        training_data:dataset.dms_dataset,
        val_data:dataset.dms_dataset,
        loss_fn,
        optimizer,
        scheduler,
        total_epochs:int,
        train_batch_size:int,
        val_batch_size:int,
        verbose:bool=False
        )->torch.nn.Module:
    
    if not verbose:
        bar = Bar("Training model...",max=total_epochs,suffix='%(percent).1f%% - %(eta)ds')
    for epoch in range(total_epochs):
        
        train_batches = DataLoader(training_data,batch_size=train_batch_size,shuffle=True)
        model.train()
        for batched_data in train_batches:
            optimizer.zero_grad()
            x = batched_data[0]
            if len(x)==1:continue #Skip batch if it is a single item
            prediction = model(x)
            loss = loss_fn(prediction,batched_data[1].to(model.get_device()))
            loss.backward()
            optimizer.step()
        
        val_loss = get_loss(model,val_data,loss_fn,val_batch_size)
        scheduler.step(val_loss)
        if not verbose:
            bar.next()

        else:
            train_loss,train_corr = get_loss_corr(model,training_data,loss_fn,train_batch_size)
            val_loss,val_corr = get_loss_corr(model,val_data,loss_fn,val_batch_size)
            print("\tEP:{}. TL:{:.3f}, TC:{:.3f}, VL:{:.3f}, VC:{:.3f}, LR:{:.2e}, Bad_Epochs:{}".format(
                epoch,
                float(train_loss),
                train_corr,
                float(val_loss),
                val_corr,
                optimizer.state_dict()['param_groups'][0]['lr'],
                scheduler.state_dict()['num_bad_epochs']
                ))

    if not verbose:bar.finish()
    return model

   
def get_model_pred(model,eval_data:dataset.dms_dataset,batch_size = 100)->torch.tensor:
    batches = DataLoader(eval_data,batch_size=batch_size,shuffle=False)
    with torch.no_grad():
        model.eval()
        predicted = []
        for batched_data in batches:
            X = batched_data[0]
            pred = model(X)
            predicted.append(pred)
    return torch.cat(predicted)

def get_loss(model,eval_data:dataset.dms_dataset,loss_fn,batch_size=100)->torch.tensor:
    predictions = get_model_pred(model,eval_data,batch_size=batch_size).cpu()
    actual = eval_data.scores
    loss = loss_fn(predictions,actual).cpu().detach()
    return loss

def get_corr(model,eval_data:dataset.dms_dataset,batch_size)->float:
    predictions = get_model_pred(model,eval_data,batch_size=batch_size).cpu()
    return float(torch.corrcoef(torch.stack((predictions,eval_data.scores)).squeeze())[0][1])

def get_loss_corr(model,eval_data:dataset.dms_dataset,loss_fn,batch_size=100)->Tuple[torch.tensor,float]:
    predictions = get_model_pred(model,eval_data,batch_size=batch_size).cpu()
    actual = eval_data.scores
    loss = loss_fn(predictions,actual).cpu().detach()
    corr = float(torch.corrcoef(torch.stack((predictions,actual)).squeeze())[0][1])
    return (loss,corr)