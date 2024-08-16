# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
esm_3_enc.py
--------------------------------------------------------------------------------
Contains class to wrap around ESM_3 protein language models.

Caches embeddings to the disk to avoid re-generating them during training

"""
import esm as esm
if esm.__version__[0]!='3':
    raise ValueError("This code needs to be run in an environment with esm3 installed")

from huggingface_hub import login
key = "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"#input("Enter huggingface key:")
login(key)

import torch
from . import tensorcache
from pathlib import Path
from typing import Tuple,List,Union
import os



from esm.pretrained import ESM3_sm_open_v0
from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
TOKENIZERS_PARALLELISM=False

root_dir = Path(__file__).resolve().parent.parent
cache_dir = os.path.join(root_dir,'.tensor_cache')


class ESM_Encoder():
    def __init__(self,encoder_name:str) -> None:
        if encoder_name.lower()!='esm3':
            raise KeyError
        self.encoder_name = "esm3"
        self.embed_dim = 1536
        self.device = 'cpu'
        self.esm=None#Will load if we need it
        self.tokenizer = EsmSequenceTokenizer()

        self.token_rep_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'token_rep')
        )
        self.averaged_rep_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'averaged_rep')
        )

    def to(self,dev: Union[str,int]):
        self.device=dev
        if self.esm!=None:
            self.esm.to(dev)

    def load_esm(self):
        self.esm=ESM3_sm_open_v0(self.device)

    def drop_esm(self):
        self.esm=None
    
    def use_ramcache(self,setting):
        self.token_rep_cache.use_ramcache(setting)
        self.averaged_rep_cache.use_ramcache(setting)
        

    def encode(self,
            seqs:List[str],
            return_reps:bool,
            return_averaged_reps:bool,
            return_token_pll:bool,
            return_total_pll:bool,
            )->Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        
        
        if isinstance(seqs,str):
            seqs=[seqs]
        if len(set(map(lambda s:len(s),seqs)))==1:#If all sequences are the same length:
            return self.encode_fl(seqs,return_reps,return_averaged_reps,return_token_pll,return_total_pll)
        else:
            return self.encode_vl(seqs,return_reps,return_averaged_reps,return_token_pll,return_total_pll)
        

    def encode_fl(self,
            seqs:List[str],
            return_reps:bool,
            return_averaged_reps:bool,
            return_token_pll:bool,
            return_total_pll:bool,
            )->Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        if isinstance(seqs,str):
            seqs=[seqs]
        need_to_encode=list()
        for seq in seqs:
            if not self.token_rep_cache.isCached(seq):
                need_to_encode.append(seq)
        if need_to_encode:
            self.add_to_cache(need_to_encode)

        if return_reps:
            token_rep = torch.stack(list(map(self.token_rep_cache.read,seqs))).to(self.device)
        else:
            token_rep = None

        if return_averaged_reps:
            averaged_rep = torch.stack(list(map(self.averaged_rep_cache.read,seqs))).to(self.device)
        else:averaged_rep=None

        if return_token_pll:
            raise NotImplementedError
            #token_pll = torch.stack(list(map(self.token_pll_cache.read,seqs))).to(self.device).unsqueeze(-1)
        else:
            token_pll = None


        if return_total_pll:
            raise NotImplementedError
            #total_pll = torch.stack(list(map(self.total_pll_cache.read,seqs))).to(self.device)
        else:
            total_pll=None
        
        return token_rep,averaged_rep,token_pll,total_pll
    
    def encode_vl(self,
            seqs:List[str],
            return_reps:bool,
            return_averaged_reps:bool,
            return_token_pll:bool,
            return_total_pll:bool,
            )->Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        
        sequences_by_length = dict()
        for (l,seq) in set(map(lambda s:(len(s),s),seqs)):
            if not self.token_rep_cache.isCached(seq):
                if l not in sequences_by_length:
                    sequences_by_length[l]=list()
                sequences_by_length[l].append(seq)
        for l in sequences_by_length:
            self.add_to_cache(sequences_by_length[l])
        max_length = max(map(lambda s:len(s),seqs))

        if return_token_pll or return_reps:
            token_reps = list()
            token_plls = list()
            for s in seqs:
                if return_reps:
                    rep = self.token_rep_cache.read(s)
                if return_token_pll:
                    raise NotImplementedError
                    #token_pll = self.token_pll_cache.read(s)
                
                sequence_length = len(rep)
                if sequence_length != max_length:
                    #Pad takes 2N arguments, where the first two are the padding for the last dimension in the front and back
                    #Then the next two are the padding for the second to last dimension, front and back
                    if return_reps:rep = torch.nn.functional.pad(rep,(0,0,0,max_length - sequence_length))
                    if return_token_pll:
                        raise NotImplementedError
                        #token_pll = torch.nn.functional.pad(token_pll,(0,max_length - sequence_length))

                if return_reps:
                    token_reps.append(rep)
                if return_token_pll:
                    raise NotImplementedError
                    #token_plls.append(token_pll)
        

        if return_reps:
            token_rep = torch.stack(token_reps).to(self.device)
        else:
            token_rep = None

        if return_averaged_reps:
            averaged_rep = torch.stack(list(map(self.averaged_rep_cache.read,seqs))).to(self.device)

        else:
            averaged_rep=None

        if return_token_pll:
            raise NotImplementedError
            #token_pll = torch.stack(token_plls).to(self.device).unsqueeze(-1)
        else:
            token_pll = None

        if return_total_pll:
            raise NotImplementedError
            #total_pll = torch.stack(list(map(self.total_pll_cache.read,seqs))).to(self.device)
        else:
            total_pll=None
        
        return token_rep,averaged_rep,token_pll,total_pll
    

    def add_to_cache(self,seqs:List[str]):
        """
        Adds a list of sequence to the caches
        All sequences must be the same length!
        """
        def seq_batch(seqs:List[str],n):
            for i in range(0,len(seqs),n):
                yield seqs[i:i+n]


        if self.esm==None:
            self.load_esm()
        for batch in seq_batch(seqs,5):
            tokenized_seqs = torch.stack(
                tuple(
                    map(
                        lambda x:torch.tensor(self.tokenizer.encode(x)),
                        batch
                        )
                )
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.esm.forward(sequence_tokens = tokenized_seqs)
                token_reps = model_output.embeddings 
                averaged_rep = torch.sum(token_reps,dim=1)
                for i in range(len(batch)):
                    self.token_rep_cache.write(batch[i],token_reps[i])
                    self.averaged_rep_cache.write(batch[i],averaged_rep[i])
    


