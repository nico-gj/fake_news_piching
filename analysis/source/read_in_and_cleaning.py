## Python Setup
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

def load_data_and_clean():
    ## Data Read In

    df = pd.read_csv('data/raw_kaggle_data/data.csv')
    df.columns = [col.lower() for col in list(df)]

    ## Data Cleaning Headline
    df['headline'] = df['headline'].str.replace(r'\(.+\)', '')
    df['headline'] = df['headline'].str.replace(r'\n', ' ')
    df['headline'] = df['headline'].str.replace(r'\W+|\d+', ' ')
    df['headline'] = df['headline'].str.strip()
    df['headline'] = df['headline'].str.lower()

    # Special Text Sequences
    df['body'] = df['body'].str.replace(r'^[A-Z\s]+\s\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'\bFILE\sPHOTO\:\s.+\n\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'\(Reuters\)\s\-\s', '')
    df['body'] = df['body'].str.replace(r'____', '')
    df['body'] = df['body'].str.replace(r'\d{4}\/\d{2}\/[A-z]+', '')
    df['body'] = df['body'].str.replace(r'https?:(\/)?\/[A-z\-\.\/]+\/([A-z\-\.\/]+)?', '')
    df['body'] = df['body'].str.replace(r'www\.[A-z\-\.\/]+\.(com|org|net)', '')
    # Classic Cleaning
    df['body'] = df['body'].str.replace(r'\(.+\)', '')
    df['body'] = df['body'].str.replace(r'\n', ' ')
    # Remove all alpha-numeric characters
    df['body'] = df['body'].str.replace(r'\W+|\d+', ' ')
    # Remove double spaces
    df['body'] = df['body'].str.replace(r'\s+', ' ')
    df['body'] = df['body'].str.strip()
    df['body'] = df['body'].str.lower()


    # Source
    df['source_1'] = df['urls'].str.extract(r'https?\:\/\/[A-z]+\.([A-z\-]+)\.')
    df['source_2'] = df['urls'].str.extract(r'https?\:\/\/([A-z\-]+)\.')
    df['source'] = np.where(df['source_1'].notnull(), df['source_1'], df['source_2'])
    del df['source_1'], df['source_2']
    df['source'] = df['source'].str.upper()

    # Restrict Dict
    df = df[(df['body'].notnull())&(df['body']!="")&(df['headline'].notnull())&(df['headline']!="")]
    df = df.reset_index()
    df.rename(columns={'index':'original_index'}, inplace=True)

    ## Export
    df_dict = df.to_dict()
    df_dict = get_dico_by_id(df_dict, 10)

    return df_dict

def get_dico_by_id(dico, body_threshold):
    new_dict = {}
    for i, id in enumerate(dico["headline"].keys()):
        data_id = retrieve_specific_data_from_id(dico, id)
        if len(data_id["body"])>body_threshold:
            new_dict[i] = data_id
    return new_dict

def retrieve_specific_data_from_id(in_dict, id):
    headline = in_dict['headline'][id].split(" ")
    label = in_dict['label'][id]
    if isinstance(in_dict["body"][id], float):
        body = [""]
    else:
        body = in_dict["body"][id].split(" ")
    return {'headline':headline, 'label':label, 'body':body}

def get_all_headlines(data):
    return [data[keys]["headline"] for keys in data.keys()]

def get_all_labels(data):
    return [data[keys]["label"] for keys in data.keys()]

def get_all_bodies(data):
    return [data[keys]["body"] for keys in data.keys()]
