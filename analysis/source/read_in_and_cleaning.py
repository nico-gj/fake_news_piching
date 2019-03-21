## Python Setup

import pandas as pd
import math
from tqdm import tqdm

def load_data_and_clean():
    ## Data Read In

    df = pd.read_csv('data/raw_kaggle_data/data.csv')
    df.columns = [col.lower() for col in list(df)]
    # df.head()
    # df.shape


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
    df['body'] = df['body'].str.lower()

    df['source'] = df['urls'].str.extract(r'https?\:\/\/www\.([A-z\-])\.')

    ## Export
    df_dict = df.to_dict()

    ## Sandbox
    # n=234
    # print(df['body'][n])
    # print('\n----------\n')
    # print(df['body'][n])
    return df_dict


def retrieve_specific_data_from_id(data, id):
    return {"headline":data["headline"][id], "body":data["body"][id], "label":data["label"][id]}


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
