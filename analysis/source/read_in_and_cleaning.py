## Python Setup
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import glob
import re
import json

def read_in_proquest(path):

    files = glob.glob('data/raw/{}/*.txt'.format(path))

    keys = ['Author', 'Full text', 'Title', 'Publication title', 'Publication year']
    out_dict = {key: [] for key in keys}

    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as in_file:
            txt = in_file.read()
        articles = txt.split('____________________________________________________________')
        del articles[len(articles)-1], articles[0]
        for article in articles:
            temp_dict = {}
            elems = article.split('\n\n')
            del elems[len(elems)-1], elems[0]
            elems = [i for i in elems if ": " in i]
            for elem in elems:
                temp_dict[elem.split(": ")[0]]=re.sub(r'^(.*?):\s', '', elem).strip()
            temp_dict = {key: temp_dict[key] if key in temp_dict.keys() else np.nan for key in keys}
            for key in keys:
                out_dict[key].append(temp_dict[key])

    df = pd.DataFrame.from_dict(out_dict)
    df = df.drop_duplicates()
    for var in list(df):
        df = df[df[var].notnull()]
    df = df.drop_duplicates('Full text')

    df.rename(columns={'Author':'author', 'Full text':'body', 'Title':'headline', 'Publication title':'source', 'Publication year':'year'}, inplace=True)

    return df


def load_data_and_clean(body_threshold=10, data='proquest'):

    if data == 'kaggle':

        ## Data Read In
        df = pd.read_csv('data/raw/kaggle_data/data.csv')
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
        df = df[df['body'].str.split(' ').apply(lambda x: len(x)>body_threshold)]
        df = df.reset_index()
        df.rename(columns={'index':'original_index'}, inplace=True)

    if data == 'proquest':

        ## Data Read In
        df = read_in_proquest(path='20190326_proquest_query')

        ## Data Cleaning Headline
        df['headline'] = df['headline'].str.replace(r'\(.+\)', '')
        df['headline'] = df['headline'].str.replace(r'\n', ' ')
        df['headline'] = df['headline'].str.replace(r'\W+|\d+', ' ')
        df['headline'] = df['headline'].str.strip()
        df['headline'] = df['headline'].str.lower()

        df['body_raw'] = df['body']

        ## Data Cleaning Body
        # Special Text Sequences
        df['body'] = df['body'].str.replace(r'\n----------\n(\n|.)+$', '')
        df['body'] = df['body'].str.replace(r'\n(Credit|CREDIT)\:[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nIllustration\sCaption\:[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nIllustration\sTangled\sweb:[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\n[^\n]+can\sbe\sreached\sat\s[^\n\@]+\@[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nThis\sis\sa\smore\scomplete\sversion\sof\sthe\sstory\sthan\sthe\sone\sthat\sappeared\sin\sprint\.(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nPhotograph\sFrom\s((Far|Top)\s)?(Above|Left|Right|Below|(t|T)op)[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nThis\slist\swas\scompiled\swith the\sassistance\sof[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nAll\sdates\sare\ssubject\sto change\.(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\n[^\n]+IS\sA\s[^\n]+\sSPECIAL\sCONTRIBUTOR\.\s[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\n[a-z\.]+@[a-z\.]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\n--?\s[^\n]+$', '')

        # Classic Cleaning
        df['body'] = df['body'].str.replace(r'[A-z\.\-]+\@[A-z\.\-]+', '') # Email address
        df['body'] = df['body'].str.replace(r'\@[A-z\-\_]+', '') # Twitter handle
        df['body'] = df['body'].str.replace(r'\([^\)]+\)', '')
        df['body'] = df['body'].str.replace(r'\n', ' ')
        # Remove all alpha-numeric characters
        df['body'] = df['body'].str.replace(r'\W+|\d+', ' ')
        # Remove double spaces
        df['body'] = df['body'].str.replace(r'\s+', ' ')
        df['body'] = df['body'].str.strip()
        df['body'] = df['body'].str.lower()

        ## Data Cleaning Source
        df['source'] = df['source'].str.replace(r'\;(.*)', '')
        df['source'] = df['source'].str.replace(r', La\s?te Ed\s?ition \(East Coast\)', '')
        df['source'] = df['source'].str.strip()
        # Map to Clean Names
        df['source'] = df['source'].str.replace(r'\s', '').str.upper()
        clean_names = pd.DataFrame({
            'source':['CHICAGOTRIBUNE', 'NEWYORKTIMES', 'NEWYORKDAILYNEWS', 'LOSANGELESTIMES', 'NEWYORKPOST', 'THEWASHINGTONPOST'],
            'clean_source':['CHICAGO TRIBUNE', 'NY TIMES', 'NY DAILY NEWS', 'LA TIMES', 'NY POST', 'WASHINGTON POST'],
            'label':[1, 1, 0, 1, 1, 0]
        })
        df = pd.merge(df, clean_names, how='left', on='source')
        del df['source']
        df.rename(columns={'clean_source':'source'}, inplace=True)

        ## Data Cleaning Year
        df['year'] = df['year'].str.replace(r'\s', '')
        df['year'] = df['year'].str.strip()
        ## Year Cat:
        n = 5   # Every how many years?
        df['year_cat'] = df['year'].apply(lambda x: int(pd.to_numeric(x, errors='coerce')/n)*n)
        df['year_cat'] = df['year_cat'].apply(lambda x: "{}-{}".format(x, x+(n-1)))
        df['year'] = "Y"+df['year']

        # Restrict Data by Removing Empty
        for var in list(df):
            df = df[(df[var].notnull())]
        df = df[df['body'].str.split(' ').apply(lambda x: len(x)>body_threshold)]
        df = df.drop_duplicates('body')

        # Restrict to news really about Kanye West
        n = 2   # How many times does he have to be mentionned?
        df = df[df['body'].apply(lambda x: len(re.findall(r'\b(kanye\swest|kanye|west|ye|yeezy)\b', x)))>=n]


        # Save Original Index Column
        df = df.reset_index()
        df.rename(columns={'index':'original_index'}, inplace=True)

    ## Export
    df_dict = df.to_dict()

    return df_dict

def get_dico_by_id(dico):
    new_dict = {}
    for i, id in enumerate(dico["headline"].keys()):
        data_id = retrieve_specific_data_from_id(dico, id)
        new_dict[i] = data_id
    return new_dict

def print_data_by_id(df_dict, i):
    for key in df_dict.keys():
        print(key)
        print(df_dict[key][i])
        print('')

def retrieve_specific_data_from_id(in_dict, id):
    headline = in_dict['headline'][id].split(" ")
    body = in_dict["body"][id].split(" ")
    source = in_dict["source"][id]
    return {'headline':headline, 'source':source, 'body':body}

def get_all_headlines(data):
    return [data[keys]["headline"] for keys in data.keys()]

def get_all_labels(data):
    return [data[keys]["label"] for keys in data.keys()]

def get_all_source(data):
    return [data[keys]["source"] for keys in data.keys()]

def get_all_bodies(data):
    return [data[keys]["body"] for keys in data.keys()]

def get_all_var(data, var):
    return [data[keys][var] for keys in data.keys()]
