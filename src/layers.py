# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
layers.py
--------------------------------------------------------------------------------
"""

from typing import Iterable, Union

import torch


class Dense(torch.nn.Module):
    """
    A Module that receives input x and does:
        y = linear combination of x
        y = activation function of y
        y = dropout y
        returns the concatenation of x and y
    """

    def __init__(
        self,
        inp : int,
        out : int,
        act : torch.nn.Module,
        drp : float):

        super().__init__()

        self._nrm = torch.nn.BatchNorm1d(inp)
        self._lin = torch.nn.Linear(inp, out)
        self._act = act
        self._drp = torch.nn.Dropout(drp)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (x,self._drp(self._act(self._lin(self._nrm(x))))),-1)


class DenseBlock(torch.nn.Module,):
    """
    Wraps a sequence of Dense modules to more
    conveniently handle their input and output dimensions.
    """

    def __init__(
        self,
        inp : int,
        out : int,
        act : torch.nn.Module,
        lyr : int,
        drp : float,
        ):

        super().__init__()

        self._inp = inp
        self._out = out
        self._lyr = torch.nn.ModuleList(
            [Dense(
                inp=inp+(x*out),
                out=out,
                act=act(),
                drp=drp) for x in range(lyr)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for lyr in self._lyr:
            x = lyr(x)
        return x

    def getOutputDim(self) -> int:
        return self._inp + len(self._lyr) * self._out


class MLP(torch.nn.Module):
    """
    Multilayer perceptron with optional hidden layer activation dropout.
    All hidden layers have the same activation function and dropout rate.
    """

    def __init__(
        self,
        inp : int,
        hid : Union[int,Iterable[int]],
        out : int,
        act : torch.nn.Module,
        drp : float):

        super().__init__()

        if isinstance(hid, int):
            hid = [hid]

        self._inp = inp
        self._out = out
        self._act = act()
        self._drp = None
        self._drp = torch.nn.Dropout(drp)
        lyr = list()
        lyr.append(torch.nn.Linear(inp, hid[0]))
        for x in range(len(hid)-1):
            lyr.append(torch.nn.BatchNorm1d(hid[x], affine=True))
            lyr.append(torch.nn.Linear(hid[x], hid[x+1]))
        lyr.append(torch.nn.BatchNorm1d(hid[-1], affine=True))
        lyr.append(torch.nn.Linear(hid[-1], out))
        self._lyr = torch.nn.ModuleList(lyr)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # for idx,lyr in enumerate(self._lyr):
        #     if isinstance(lyr, torch.nn.Linear):
        #         x = lyr(self._drp(x))
        #         if idx < len(self._lyr) - 1:
        #             x = self._act(x)
        #     else:
        #         x = lyr(x)
        # return x
        if x.ndim==1:
            x=x.reshape(1,-1)
        for idx,lyr in enumerate(self._lyr):
            x = lyr(x)
            if isinstance(lyr, torch.nn.Linear) and \
               idx < len(self._lyr) - 1:
                x = self._drp(self._act(x))
        return x

    def getOutputDim(self) -> int:
        return self._out
