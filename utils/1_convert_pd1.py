import pickle
import pandas as pd

folder_path = '../raw_data/'
data = 'reviews_CDs_and_Vinyl_5.json'
meta_data = 'meta_CDs_and_Vinyl.json'

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

reviews_df = to_df(folder_path + data)
with open('../raw_data/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df(folder_path + meta_data)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('../raw_data/meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
