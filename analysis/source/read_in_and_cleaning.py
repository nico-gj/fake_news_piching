## Python Setup

import pandas as pd
from text_analysis import create_bag_of_words, create_topics, get_word_counts


## Data Read In

df = pd.read_csv('../data/raw_kaggle_data/data.csv')
df.columns = [col.lower() for col in list(df)]
# df.head()
# df.shape


## Data Cleaning

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
