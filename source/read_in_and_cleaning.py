## Python Setup
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import glob
import re
import json
import nltk
nltk.download('wordnet')

def classic_var_cleaning(df, var):
    df[var] = df[var].str.replace(r'\([^\)]+\)', ' ')
    df[var] = df[var].str.replace(r'\[[^\]]+\]', ' ')
    df[var] = df[var].str.replace(r'\n', ' ')
    df[var] = df[var].str.replace(r'\.', ' ') # Replace periods by empty (D.C., for example)
    df[var] = df[var].str.replace(r'\W+|\d+', ' ') # Remove all alpha-numeric characters
    df[var] = df[var].str.replace(r'\s+', ' ') # Remove double spaces
    df[var] = df[var].str.strip()
    df[var] = df[var].str.lower()
    return df[var]

def read_in_and_clean(min_body_threshold=0.1, max_body_threshold=0.95, data='george_mcintyre', extra_path="", fake_subset=None):

    if data == 'kaggle':

        ## Data Read In
        df = pd.read_csv(extra_path+'data/raw/kaggle_data/data.csv')
        df.columns = [col.lower() for col in list(df)]

        ## Data Cleaning Headline
        df['headline'] = classic_var_cleaning(df, 'headline')

        # Special Text Sequences
        df['body'] = df['body'].str.replace(r'^[A-Z\s]+\s\(Reuters\)\s\-\s', '')
        df['body'] = df['body'].str.replace(r'\bFILE\sPHOTO\:\s.+\n\(Reuters\)\s\-\s', '')
        df['body'] = df['body'].str.replace(r'\(Reuters\)\s\-\s', '')
        df['body'] = df['body'].str.replace(r'____', '')
        df['body'] = df['body'].str.replace(r'\d{4}\/\d{2}\/[A-z]+', '')
        df['body'] = df['body'].str.replace(r'https?:(\/)?\/[A-z\-\.\/]+\/([A-z\-\.\/]+)?', '')
        df['body'] = df['body'].str.replace(r'www\.[A-z\-\.\/]+\.(com|org|net)', '')
        # Classic Cleaning
        df['body'] = classic_var_cleaning(df, 'body')

        # Source
        df['source_1'] = df['urls'].str.extract(r'https?\:\/\/[A-z]+\.([A-z\-]+)\.')
        df['source_2'] = df['urls'].str.extract(r'https?\:\/\/([A-z\-]+)\.')
        df['source'] = np.where(df['source_1'].notnull(), df['source_1'], df['source_2'])
        del df['source_1'], df['source_2']
        df['source'] = df['source'].str.upper()

        # Restrict Dict
        df = df[(df['body'].notnull())&(df['body']!="")&(df['headline'].notnull())&(df['headline']!="")]
        df = df[df['body'].apply(lambda x: len(x.split(' '))>=df['text'].apply(lambda x: len(x.split(' '))).quantile(min_body_threshold))]
        df = df.reset_index()
        df.rename(columns={'index':'id'}, inplace=True)

    if data == 'george_mcintyre':

        ## Data Read In
        df = pd.read_csv(extra_path+'data/raw/george_mcintyre/fake_or_real_news.csv')
        df.rename(columns={'title': 'headline', 'text': 'body'}, inplace=True)

        # temp:
        df['body_raw'] = df['body']
        df['headline_raw'] = df['headline']

        ## Convert label to binary
        df['label'] = [1 if x=="FAKE" else 0 for x in df['label']]

        ## Source
        df['source'] = "Unknown"

        ## Data Cleaning Headline
        # Special Text Cleaning;
        df['headline'] = df['headline'].str.replace(r'\s-\sThe Onion\s-\sAmerica\'s\sFinest\sNews\sSource$', '')
        df['headline'] = df['headline'].str.replace(r'\s·\sGuardian\sLiberty\sVoice$', '')
        # Classic Cleaning
        df['headline'] = classic_var_cleaning(df, 'headline')

        ## Data Cleaning Body
        # Special Text Sequences
        df['body'] = df['body'].str.replace(r'\n', '\n\n')
        df['body'] = df['body'].str.replace(r'Advertisement\s-\sstory\scontinues\sbelow\b', '')
        df['body'] = df['body'].str.replace(r'^BNI\sStore (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d\d?\s\d{4}\b', '')
        df['body'] = df['body'].str.replace(r'(Edited|Written) by [A-z\s\.]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\nSources:\n(.|\n)+$', '\n')
        df['body'] = df['body'].str.replace(r'\n(Top\sand\sFeatured|First\sInline)\sImages?\sCourtesy\sof\s[^\n]+(\n|$)', '\n')
        df['body'] = df['body'].str.replace(r'\s\|\sInfowars\.com(\s+)?\n', ' ')
        df['body'] = df['body'].str.replace(r'\bGet\sthe\slatest\sbreaking\snews\s\&\sspecials\sfrom\sAlex\sJones\sand\sthe\sInfowars\sCrew\..+$', ' ')
        df['body'] = df['body'].str.replace(r'(https?:(\/)?\/)?((www\.|pic\.))?[A-z]+\.([a-z]{2})[^\s]+', ' ') # Remove websites
        df['body'] = df['body'].str.replace(r'^\d+(\sViews)?(\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d\d?\,\s\d{4})?\s+GOLD(\s+)?\,\sKWN\sKing\sWorld\sNews(\s+)?\n', ' ')
        df['body'] = df['body'].str.replace(r'\n\*\*\*ALSO\sJUST\sRELEASED\:[^\n]+(\n|$)', ' ')
        df['body'] = df['body'].str.replace(r'(^|\n)©[^\n]+(\n|$)', ' ')
        df['body'] = df['body'].str.replace(r'(^|\n)Print(\s+)?(\n|$)', ' ')
        # Classic Cleaning
        df['body'] = classic_var_cleaning(df, 'body')

        ## Restrict Dict
        df = df[(df['body'].notnull())&(df['body']!="")&(df['headline'].notnull())&(df['headline']!="")]
        # Restrict by length of body:
        min_len = df['body'].apply(lambda x: len(x.split(' '))).quantile(min_body_threshold)
        max_len = df['body'].apply(lambda x: len(x.split(' '))).quantile(max_body_threshold)
        df = df[df['body'].apply(lambda x: len(x.split(' '))>=min_len)]
        df = df[df['body'].apply(lambda x: len(x.split(' '))<=max_len)]

    if data=='isot':

        ## Data Read In
        df1 = pd.read_csv(extra_path+'data/raw/isot_victoria/True.csv')
        df1['label'] = 0
        df2 = pd.read_csv(extra_path+'data/raw/isot_victoria/Fake.csv')
        df2['label'] = 1
        df = pd.concat([df1, df2])
        del df['subject']
        df.rename(columns={'title':'headline', 'text':'body'}, inplace=True)

        df['headline'] = classic_var_cleaning(df, 'headline')

        df['body'] = df['body'].str.replace(r'(\d{4})([A-z])', r'\1 \2')
        df['body'] = df['body'].str.replace(r'Featured\simage\svia\s[A-z\/\s]+$', r'')
        df['body'] = df['body'].str.replace(r'Via\:\s[A-z\/\s]+$', r'')
        df['body'] = classic_var_cleaning(df, 'body')

    if fake_subset==0:
        df = df[df['label']==0].reset_index(drop=True)
    if fake_subset==1:
        df = df[df['label']==1].reset_index(drop=True)

    ## Export
    df_dict = df.to_dict()

    return df_dict

def retrieve_word_seq_text(df_dict, text_var, lemmatize=False):
    texts = [str.split(" ") for str in list(df_dict[text_var].values())]

    if lemmatize==True:
        lemma = nltk.wordnet.WordNetLemmatizer()
        for text in texts:
            for i in range(len(text)):
                text[i] = lemma.lemmatize(text[i])

    return texts

def print_data_by_id(df_dict, i):
    for key in df_dict.keys():
        print(key)
        print(df_dict[key][i])
        print('')