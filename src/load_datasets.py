# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
OptiProt Academic License
load_datasets.py
--------------------------------------------------------------------------------
"""

import os

from pathlib import Path

import pandas as pd

from . import dataset

root_dir = Path(__file__).resolve().parent.parent


def load_c1()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'single_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )


def load_r2()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'random_doubles_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )

def load_r10()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'random_up_to_ten_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )

def load_a39()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'additive_designed_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )


def load_lr_c1r2()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'lr_c1r2_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )

def load_cnn_c1r2()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'cnn_c1r2_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )

def load_rnn_c1r2()->dataset.dms_dataset:
    df = pd.read_csv(
        os.path.join(
            root_dir,
            'data',
            'rnn_c1r2_processed.csv'
        )
    )
    df=df[df['viral_selection']!=float('-inf')]
    df=df[df['viral_selection']!=float('inf')]
    return dataset.dms_dataset(
        seqs = df['full_sequence'].to_list(),
        scores = df['viral_selection'].to_list(),
        is_viable=df['is_viable'].to_list()
    )
