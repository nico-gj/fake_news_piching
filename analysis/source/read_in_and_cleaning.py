## Python Setup

import pandas as pd
import math
from tqdm import tqdm

def load_data_and_clean():
    ## Data Read In

    df = pd.read_csv('data/raw_kaggle_data/data.csv')
    df.columns = [col.lower() for col in list(df)]

    ## Data Cleaning Headline
    df['headline'] = df['headline'].str.replace(r'\'', '')
    df['headline'] = df['headline'].str.replace(r'\"', '')
    df['headline'] = df['headline'].str.replace(r'(\,|\;|\.|\:|-|\&|\?|\'s|“|”|’|‘|\!|…|\(|\)|\[|\])', '')
    df['headline'] = df['headline'].str.lower()

    # Special Text Sequences
    df['body'] = df['body'].str.replace(r'^[A-Z\s]+\s\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'\bFILE\sPHOTO\:\s.+\n\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'____', '')

    # Classic Cleaning
    df['body'] = df['body'].str.replace(r'\(.+\)', '')
    df['body'] = df['body'].str.replace(r'\s+', ' ')
    df['body'] = df['body'].str.replace(r'\n', ' ')
    df['body'] = df['body'].str.replace(r'\'', '’')
    df['body'] = df['body'].str.replace(r'  ', '')
    df['body'] = df['body'].str.replace(r'---', '')
    df['body'] = df['body'].str.replace(r'\#', '')
    df['body'] = df['body'].str.replace(r'http//wwwconservativedailynewscom', '')
    df['body'] = df['body'].str.replace(r'\-', '')
    df['body'] = df['body'].str.replace(r'wwwthedailysheeplecom', '')
    df['body'] = df['body'].str.replace(r'https?:\/[A-z\-\.\/]+\/([A-z\-\.\/]+)?', '')
    df['body'] = df['body'].str.replace(r'\d{4}\/\d{2}\/[A-z]+', '')
    df['body'] = df['body'].str.strip()
    df['body'] = df['body'].str.replace(r'(\,|\;|\.|\:|-|\&|\?|\'s|“|”|’|‘|\!|…|\(|\)|\[|\]|\-|\-)', '')
    df['body'] = df['body'].str.replace(r'\-', '')
    df['body'] = df['body'].str.replace(r'\/', '')
    df['body'] = df['body'].str.replace(r'\-', '')
    df['body'] = df['body'].str.lower()

    df['source'] = df['urls'].str.extract(r'https?\:\/\/www\.([A-z\-]+)\.')
    df['source'] = df['source'].str.upper()

    ## Export
    df_dict = df.to_dict()
    
    return df_dict


def get_dico_by_id(dico, body_threshold):
    new_dict = {}
    for id in dico.keys():
        data_id = retrieve_specific_data_from_id(dico, id)
        if not (isinstance(data_id["body"], float) or len(data_id["body"]<body_threshold)):
            new_dict[id] = data_id
    return new_dict

def retrieve_specific_data_from_id(in_dict, id):
    out_dict = {}
    for key in in_dict.keys():
        print(key)
        out_dict[key] = in_dict[key][id]
    return out_dict

def get_all_headlines(data):
    headlines = list()
    return [elem[1].split(" ") for elem in sorted(data["headline"].items())]

def get_all_labels(data):
    labels = list()
    return [elem[1] for elem in sorted(data["label"].items())]

def get_all_bodies(data):
    bodies = list()
    for elem in sorted(data["body"].items()):
        if isinstance(elem[1], float):
            bodies.append([""])
        else:
            bodies.append(elem[1].split(" "))
    return bodies
