# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
zero_shot.py
--------------------------------------------------------------------------------
"""

from pathlib import Path

import torch

from . import esm_enc

from typing import List

from progress.bar import Bar

class PLLZeroShot():
    def __init__(self,encoder_name:str):
        self.encoder_name = encoder_name
        self.device='cpu'
        self.encoder=esm_enc.ESM_Encoder(encoder_name)
    
    def to(self,dev):
        self.device=dev
        self.encoder.to(dev)

    def predict(self,seqs:List[str],batch_size=16):
        if isinstance(seqs,str):
            seqs=[seqs]
        
        plls = list()
        with Bar("Encoding...  ",max=len(seqs),suffix='%(percent).1f%% - %(eta)ds') as bar:
            for i in range(0,len(seqs),batch_size):
                bar.next(batch_size)
                sub_seqs = seqs[i:i+batch_size]
                _,_,pll = self.encoder.encode(sub_seqs)
                plls.append(pll)
            return torch.cat(plls)