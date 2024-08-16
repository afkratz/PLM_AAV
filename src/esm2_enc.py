# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
esm_enc.py
--------------------------------------------------------------------------------
Contains class to wrap around ESM protein language models.

Caches embeddings to the disk to avoid re-generating them during training

Supports extracting three different types of embeddings from ESM:
    token_rep: 
        The embedding of the AA sequence for each residue passed in.
        Shape is NxD where N is the length of the AA sequence and D is the 
        size of the PLMs embed_dim.

    token_pll:
        The psuedo-log-likelihood of predicting the input AA at each residue.
        Including these values can improve the embeddings usefuleness.
        Shape is Nx1 where N is the length of the AA sequence.          
    
    total_pll:
        The mean of the token_pll across the length of the AA sequence.
        Useful as a zero-shot indication of the "coherence" of the protein
        in the PLMs "opinion". Low values often correlate to non-functional
        proteins.
"""
import esm

if esm.__version__[0]!='2':
    raise ValueError("This code needs to be run in an environment with esm2 installed")
    
import torch
from . import tensorcache

from pathlib import Path
import os

from typing import Tuple,List,Union


root_dir = Path(__file__).resolve().parent.parent
cache_dir = os.path.join(root_dir,'.tensor_cache')

class ESM_Encoder():
    def __init__(self,encoder_name:str):
        self.encoder_name = encoder_name
        self.device='cpu'
        
        (esm,alphabet) = load_esm_model(encoder_name)

        #Since this uses caching, we only store the ESM model if we need it
        self.esm = None
        self.embed_dim = esm.embed_dim
        self.alphabet = alphabet

        self.token_rep_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'token_rep')
        )

        self.averaged_rep_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'averaged_rep')
        )

        self.token_pll_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'token_pll')
        )

        self.total_pll_cache=tensorcache.TensorCache(
            os.path.join(cache_dir,self.encoder_name,'total_pll')
        )
        

    def to(self,dev: Union[str,int]):
        self.device=dev
        if self.esm!=None:
            self.esm.to(dev)

    def load_esm(self):
        (esm,alphabet) = load_esm_model(self.encoder_name)
        self.esm = esm
        self.to(self.device)
    
    def drop_esm(self):
        self.esm=None

    def use_ramcache(self,setting):
        self.token_rep_cache.use_ramcache(setting)
        self.token_pll_cache.use_ramcache(setting)
        self.total_pll_cache.use_ramcache(setting)
        
    
    def encode(self,
               seqs:List[str],
               return_reps:bool,
               return_averaged_reps:bool,
               return_token_pll:bool,
               return_total_pll:bool,
               )->Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        """
        Encode a list of proteins
        If proteins are all of one length, uses encode_fl
        If proteins are of variable length, uses encode_vl and returns embeddings padded with zeros
        to make all embeddings equal length

        Returns (token_rep,averaged_rep,token_pll,total_pll), or None for values where the parameter is False
        """
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

        need_to_encode = list()
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
            token_pll = torch.stack(list(map(self.token_pll_cache.read,seqs))).to(self.device).unsqueeze(-1)
        else:
            token_pll = None


        if return_total_pll:
            total_pll = torch.stack(list(map(self.total_pll_cache.read,seqs))).to(self.device)
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
    
        if isinstance(seqs,str):
            seqs=[seqs]

        #Divide up the sequences that we need to encode by their length, because
        #add_to_cache requires uniform length sequences

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
                if return_reps:rep = self.token_rep_cache.read(s)
                if return_token_pll:token_pll = self.token_pll_cache.read(s)
                
                sequence_length = len(rep)
                if sequence_length != max_length:
                    #Pad takes 2N arguments, where the first two are the padding for the last dimension in the front and back
                    #Then the next two are the padding for the second to last dimension, front and back
                    if return_reps:rep = torch.nn.functional.pad(rep,(0,0,0,max_length - sequence_length))
                    if return_token_pll:token_pll = torch.nn.functional.pad(token_pll,(0,max_length - sequence_length))
                if return_reps:token_reps.append(rep)
                if return_token_pll:token_plls.append(token_pll)
        

        if return_reps:
            token_rep = torch.stack(token_reps).to(self.device)
        else:
            token_rep = None

        if return_averaged_reps:
            averaged_rep = torch.stack(list(map(self.averaged_rep_cache.read,seqs))).to(self.device)
        else:
            averaged_rep=None

        if return_token_pll:
            token_pll = torch.stack(token_plls).to(self.device).unsqueeze(-1)
        else:
            token_pll = None


        if return_total_pll:
            total_pll = torch.stack(list(map(self.total_pll_cache.read,seqs))).to(self.device)
        else:
            total_pll=None
        
        return token_rep,averaged_rep,token_pll,total_pll

    
    def add_to_cache(self,seqs:List[str]):
        """
        Adds a sequences token_reps, token_plls, and total_plls to the caches
        All sequences must be the same length!
        """
        if self.esm==None:
            self.load_esm()
        
        encoded_seqs = list(
            map(lambda s:self.alphabet.encode('<cls>'+s+'<eos>'),seqs)
            )
        
        input_tensor = torch.tensor(encoded_seqs).to(self.device)

        with torch.no_grad():
            esm_output = self.esm(
                input_tensor,
                repr_layers = [self.esm.num_layers],
                return_contacts=False
            )

        #[:,1:-1] to exclude  <cls> and <eos>
        token_reps = esm_output['representations'][self.esm.num_layers][:,1:-1]
        averaged_rep = torch.sum(token_reps,dim=1)
        logits = esm_output['logits']
        smax = torch.nn.Softmax(dim=2)
        scaled_logits = smax(logits)[:,1:-1]
        tok = input_tensor[:,1:-1]
        odds = scaled_logits[torch.arange(tok.shape[0])[:,None],torch.arange(tok.shape[1]),tok]
        token_pll = torch.log2(odds)
        total_pll = token_pll.mean(axis=1).unsqueeze(1)
        for i in range(len(seqs)):
            self.token_rep_cache.write(seqs[i],token_reps[i])
            self.averaged_rep_cache.write(seqs[i],averaged_rep[i])
            self.token_pll_cache.write(seqs[i],token_pll[i])
            self.total_pll_cache.write(seqs[i],total_pll[i])



def load_esm_model(encoder_name:str)->Tuple[esm.model.esm2.ESM2,esm.data.Alphabet]:
    encoder_name = encoder_name.lower()
    if encoder_name=="esm2_t6_8m_ur50d":
        return esm.pretrained.esm2_t6_8M_UR50D()
    if encoder_name=="esm2_t12_35m_ur50d":
        return esm.pretrained.esm2_t12_35M_UR50D()
    if encoder_name=="esm2_t30_150m_ur50d":
        return esm.pretrained.esm2_t30_150M_UR50D()
    if encoder_name=="esm2_t33_650m_ur50d":
        return esm.pretrained.esm2_t33_650M_UR50D()
    if encoder_name=="esm2_t36_3b_ur50d":
        return  esm.pretrained.esm2_t36_3B_UR50D()
    if encoder_name=="esm2_t48_15b_ur50d":
        return esm.pretrained.esm2_t48_15B_UR50D()   
    raise KeyError("Model {} not found".format(encoder_name))


encoder_names = [
    'esm2_t6_8m_ur50d',
    'esm2_t12_35m_ur50d',
    'esm2_t30_150m_ur50d',
    'esm2_t33_650m_ur50d',
    'esm2_t36_3b_ur50d',
    'esm2_t48_15b_ur50d' # Too big to run on my machine
]