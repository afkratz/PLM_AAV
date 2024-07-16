# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
averagers.py
--------------------------------------------------------------------------------
"""

import torch

from .layers import DenseBlock

from . import esm_enc

from typing import List

class SimpleAverager(torch.nn.Module):
    """
    Converts a representation of tokens into a single embedding by simply summing
    across the length of the protein. We sum instead of average to ignore zero values
    if we are doing variable length encoding with zero padding.
    Optionally, includes token_pll as a feature included in the embedding
    """
    def __init__(
            self,
            encoder_name:str,
            use_pll:bool,
            **kwargs,
                 ):
        super().__init__()
        self.encoder=esm_enc.ESM_Encoder(encoder_name)
        self.use_pll = use_pll
        self.embed_dim = self.encoder.embed_dim + int(use_pll)

    def forward(self,x:List[str])->torch.tensor:
        _,averaged_reps,_,total_pll = self.encoder.encode(x,return_reps=False,return_averaged_reps=True,return_token_pll=False,return_total_pll=self.use_pll)
        if self.use_pll:
            return torch.cat((averaged_reps,total_pll),dim=-1)
        else:
            return averaged_reps
    
    def to(self,dev):
        self.encoder.to(dev)
        
class LearnableAverager(torch.nn.Module):
    """
    Converts a representation of tokens into a single embedding by summing the embeddings
    across the length of the protein. 
    Before summing, applies a linear transformation and a ReLU to the embeddings to enable
    the model to learn to include or exclude features.
    Sums instead of averages to ignore zero-padded variable length sequences
    Optionally, includes token_pll as a feature included in the embedding before linear transform
    and includes total_pll as a feature re-added after averaging.    
    """
    def __init__(
            self,
            encoder_name:str,
            averager_output_dim:int,
            use_total_pll:bool,
            use_token_pll:bool,
            **kwargs
                 ):
        super().__init__()
        self.encoder =esm_enc.ESM_Encoder(encoder_name)
        self.use_total_pll=use_total_pll
        self.use_token_pll=use_token_pll
        in_features = self.encoder.embed_dim + int(use_token_pll)

        self.pre_average_transform = torch.nn.Linear(
            in_features=in_features,
            out_features=averager_output_dim,
            bias=False#To prevent the model from knowing how many zeros are on the tail
        )

        total_output_features = averager_output_dim + int(use_total_pll)
        self.embed_dim = total_output_features

    def to(self,dev):
        self.pre_average_transform.to(dev)
        self.encoder.to(dev)

    def forward(self,x:List[str])->torch.tensor:
        token_reps,_,token_pll,total_pll = self.encoder.encode(x,
                                                               return_reps=True,
                                                               return_averaged_reps=False,
                                                               return_token_pll=self.use_token_pll,
                                                               return_total_pll=self.use_total_pll)

        if self.use_token_pll:
            rep = torch.cat((token_reps,token_pll),dim=-1)
        else:
            rep = token_reps
        
        output = torch.sum(
            torch.nn.functional.relu(self.pre_average_transform(rep)),
            axis=1
        )
        if self.use_total_pll:
            output = torch.cat((output,total_pll),dim=-1)

        return output
    

class SimpleAveragerDense(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.sav=SimpleAverager(**kwargs)

        self.denseblock = DenseBlock(
            inp = self.sav.embed_dim,
            out = kwargs['out'],
            act = kwargs.get('act',torch.nn.ReLU),
            lyr = kwargs.get('lyr'),
            drp = kwargs.get('drp',0.0),
        )

        self.final = torch.nn.Linear(
            in_features=self.denseblock.getOutputDim(),
            out_features=1,
            bias=False
        )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x: List[str]) -> torch.tensor:
        return self.final.forward(
            self.denseblock.forward(
                self.sav.forward(x)
            )
        ).squeeze(dim=1)

    def to(self,dev):
        self.sav.to(dev)
        self.denseblock.to(dev)
        self.final.to(dev)
    
    def use_ramcache(self,setting):
        self.sav.encoder.use_ramcache(setting)

class LearnableAveragerDense(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.lav=LearnableAverager(**kwargs)

        self.denseblock = DenseBlock(
            inp = self.lav.embed_dim,
            out = kwargs['out'],
            act = kwargs.get('act',torch.nn.ReLU),
            lyr = kwargs.get('lyr'),
            drp = kwargs.get('drp',0.0),
        )

        self.final = torch.nn.Linear(
            in_features=self.denseblock.getOutputDim(),
            out_features=1,
            bias=False
        )


    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x: List[str]) -> torch.tensor:
        return self.final.forward(
            self.denseblock.forward(
                self.lav.forward(x)
            )
        ).squeeze(dim=1)

    def to(self,dev):
        self.lav.to(dev)
        self.denseblock.to(dev)
        self.final.to(dev)

    def use_ramcache(self,setting):
        self.lav.encoder.use_ramcache(setting)