# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------
Copyright 2023 Alexander Kratz [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
generate_dataset.py
--------------------------------------------------------------------------------
"""
import os
from pathlib import Path
import requests
import pandas as pd
from Bio import SeqIO

root_dir = Path(__file__).resolve().parent.parent
data_dir=os.path.join(root_dir,'data')


def download_supp_data():
    save_path = os.path.join(data_dir,'41587_2020_793_MOESM3_ESM.csv')
    if os.path.exists(save_path):return
    url = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-020-00793-4/MediaObjects/41587_2020_793_MOESM3_ESM.csv'
    
    
    print("Downloading supp table from \n \"Deep diversification of an AAV capsid protein by machine learning\"\n by Bryant, Bashir [...] & Kelsic")
    response = requests.get(url)
    if response.status_code==200:
        with open(save_path,'wb') as fh:
            fh.write(response.content)

def load_supp_data()->pd.DataFrame:
    return pd.read_csv(
        os.path.join(data_dir,'41587_2020_793_MOESM3_ESM.csv')
    )

def load_ref_seq()->str:
    return str(
        SeqIO.read(
            os.path.join(
                root_dir,'data','P03125.fasta'
            ),'fasta'
        ).seq
    )

def apply_mutation_string(mut_string:str,ref_seq:str)->str:
    return ref_seq[:560]+mut_string.upper()+ref_seq[588:]

def make_data_sets():
    if not os.path.exists(data_dir):os.mkdir(data_dir)
    download_supp_data()
    data_set = load_supp_data()
    ref_seq = load_ref_seq()
    for category in ['single',]:
        sub_df = data_set[data_set['partition']==category]
        print(category,sub_df)
        sub_df['full_sequence'] = sub_df['sequence'].apply(lambda seq:apply_mutation_string(seq,ref_seq))
        sub_df.to_csv(
            os.path.join('data','{}_processed.csv'.format(category)),index=False
        )


if __name__=="__main__":
    make_data_sets()