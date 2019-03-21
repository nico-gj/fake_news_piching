## Python Setup

import pandas as pd


def load_data_and_clean():
    ## Data Read In

    df = pd.read_csv('data/raw_kaggle_data/data.csv')
    df.columns = [col.lower() for col in list(df)]
    # df.head()
    # df.shape


    ## Data Cleaning Headline
    df['headline'] = df['headline'].str.replace(r'\'', '')
    df['headline'] = df['headline'].str.replace(r'\"', '')
    df['headline'] = df['headline'].str.replace(r'(\,|\;|\.|\:|-|\&|\?|\'s|“|”|’|‘|\!|…|\(|\))', '')
    #df['headline'] = df['headline'].str.replace(r'\;', '')
    #df['headline'] = df['headline'].str.replace(r'\.', '')
    #df['headline'] = df['headline'].str.replace(r'\:', '')
    #df['headline'] = df['headline']


    ## Data Cleaning Body 
    df['body_c'] = df['body']

    # Special Text Sequences
    df['body_c'] = df['body_c'].str.replace(r'^[A-Z\s]+\s\(Reuters\)\s\-\s', '')
    df['body_c'] = df['body_c'].str.replace(r'\bFILE\sPHOTO\:\s.+\n\(Reuters\)\s\-\s', '')
    df['body_c'] = df['body_c'].str.replace(r'\(Reuters\)\s\-\s', '')

    # Classic Cleaning
    df['body_c'] = df['body_c'].str.replace(r'\(.+\)', '')
    df['body_c'] = df['body_c'].str.replace(r'\s+', ' ')
    df['body_c'] = df['body_c'].str.replace(r'\n', ' ')
    df['body_c'] = df['body_c'].str.replace(r'\'', '’')
    df['body_c'] = df['body_c'].str.strip()


    ## Export
    df_dict = df.to_dict()

    ## Sandbox
    # n=234
    # print(df['body'][n])
    # print('\n----------\n')
    # print(df['body_c'][n])
    return df_dict


def retrieve_specific_data_from_id(data, id):
    return {"headline":data["headline"][id], "body":data["body_c"][id], "label":data["label"][id]}


def get_all_headlines(data):
    headlines = list()
    return [elem[1] for elem in sorted(data["headline"].items())]
        
def get_all_labels(data):
    labels = list()
    return [elem[1] for elem in sorted(data["label"].items())]