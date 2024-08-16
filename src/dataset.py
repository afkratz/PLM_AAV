# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
dataset.py
--------------------------------------------------------------------------------
"""

import torch
from torch.utils.data import Dataset
from typing import List,Tuple
import copy

class dms_dataset(Dataset):
    def __init__(self,seqs:List[str],scores:List[float],is_viable:List[bool]):
        if len(seqs) != len(scores):
            raise ValueError("seq len {} != score len {}".format(
                len(seqs),
                len(scores)
            ))
        if len(seqs) != len(is_viable):
            raise ValueError("seq len {} != is_viable len {}".format(
                len(seqs),
                len(is_viable)
            ))
        self.seqs = seqs
        if isinstance(scores,list):
            self.scores=torch.tensor(scores)
        elif isinstance(scores,torch.Tensor):
            self.scores=scores
        else:
            raise TypeError
        
        self.is_viable = is_viable
        
    def __len__(self)->int:
        return len(self.seqs)
    
    def __getitem__(self,idx)->Tuple[str,torch.tensor]:
        X=self.seqs[idx]
        y=self.scores[idx]
        return (X,y)


    def copy(self)->'dms_dataset':
        return copy.deepcopy(self)
    
    def split_at_n(self,idx)->Tuple['dms_dataset','dms_dataset']:
        a = copy.deepcopy(dms_dataset(seqs = self.seqs[:idx],
                                      scores = self.scores[:idx].clone().detach(),
                                      is_viable = self.is_viable[:idx]
                                      ))
        
        b = copy.deepcopy(dms_dataset(seqs = self.seqs[idx:],
                                      scores = self.scores[idx:].clone().detach(),
                                      is_viable = self.is_viable[idx:]
                                      ))
        return (a,b)
    
    def split_by_modulo(self,divisor:int,remainder:int)->Tuple['dms_dataset','dms_dataset']:
        a_seqs = list()
        a_scores = list()
        a_is_viable = list()

        b_seqs = list()
        b_scores=list()
        b_is_viable = list()

        for i in range(len(self)):
            if i%divisor == remainder:
                a_seqs.append(self.seqs[i])
                a_scores.append(self.scores[i])
                a_is_viable.append(self.is_viable[i])
            else:
                b_seqs.append(self.seqs[i])
                b_scores.append(self.scores[i])
                b_is_viable.append(self.is_viable[i])
        a = dms_dataset(seqs=a_seqs,scores=a_scores,is_viable=a_is_viable)
        b = dms_dataset(seqs=b_seqs,scores=b_scores,is_viable=b_is_viable)
        return (a,b)
    
    def extend(self,other:'dms_dataset'):
        self.seqs.extend(other.seqs)
        self.scores=torch.cat((self.scores,other.scores))
        self.is_viable.extend(other.is_viable)

    def shuffle(self,seed:int = None):
        """
        Shuffles the dataset using the random seed passed to it,
        or if no seed passed use system time to randomize
        maintains pairing between seqs and scores
        """

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(torch.initial_seed())

        # Generate random permutation of indices
        indices = torch.randperm(len(self.seqs))

        # Shuffle seqs and scores based on the random permutation
        self.seqs = [self.seqs[i] for i in indices]
        self.is_viable  = [self.is_viable[i] for i in indices]
        self.scores = self.scores[indices]
    
    def take_fraction(self,fraction:float)->'dms_dataset':
        index = int(fraction*len(self))
        sub,_ = self.split_at_n(index)
        return sub