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
    

def load_ref_seq()->str:
    return str(
        SeqIO.read(
            os.path.join(
                root_dir,'data','P03125.fasta'
            ),'fasta'
        ).seq
    )


def make_data_set():
    if not os.path.exists(data_dir):os.mkdir(data_dir)
    




if __name__=="__main__":
    make_data_set()