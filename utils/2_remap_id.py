import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import time
from copy import deepcopy

random.seed(1234)
data_dir = '../raw_data/'
Data = '../Data/'

with open(data_dir + 'reviews.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  # reviews_df = reviews_df[:300000]
  reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
  reviews_df['unixReviewTime'] = reviews_df['unixReviewTime'] // 3600 // 24
with open(data_dir + 'meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  meta_df = meta_df[['asin', 'categories']]
  meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


# fore-deal
def remove_infrequent_items(df, min_counts=5):
  counts = df['asin'].value_counts()
  df = df[df["asin"].isin(counts[counts >= min_counts].index)]
  print("items with < {} interactoins are removed".format(min_counts))
  return df

def remove_infrequent_users(df, min_counts=10):
  counts = df['reviewerID'].value_counts()
  df = df[df["reviewerID"].isin(counts[counts >= min_counts].index)]
  print("users with < {} interactoins are removed".format(min_counts))
  return df

# select user session >= 4
def select_sessions(df, mins, maxs):
  users = df['reviewerID'].unique()
  counter = 0
  allcount = len(users)
  selected_id = []
  for reviewerID, group in df.groupby('reviewerID'):
    counter += 1
    time_len = len(group['unixReviewTime'].unique())
    if time_len >= mins and time_len <= maxs:
      selected_id.append(reviewerID)

    sys.stdout.write('Session select: {:.2f}%\r'.format(100 * counter / allcount))
    sys.stdout.flush()
    time.sleep(0.01)
  df = df[df['reviewerID'].isin(selected_id)]
  print('selected session({0} <= session <= {1}):{2}'.format(mins, maxs, len(df)))
  return df

# select from meta_df
def select_meta(df, meta_df):
  items = df['asin'].unique()
  return meta_df[meta_df['asin'].isin(items)]

reviews_df = remove_infrequent_users(reviews_df, 10)
reviews_df = remove_infrequent_items(reviews_df, 8)
reviews_df = select_sessions(reviews_df, 4, 90)
meta_df = select_meta(reviews_df, meta_df)
print('num of users:{}, num of items:{}'.format(len(reviews_df['reviewerID'].unique()), len(reviews_df['asin'].unique())))
print('Select all done.')


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key


asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))


meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)

item_cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
item_cate_list = np.array(item_cate_list, dtype=np.int32)


with open(Data + 'remap.pkl', 'wb') as f:
  pickle.dump((reviews_df, meta_df), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(item_cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count, example_count), f, pickle.HIGHEST_PROTOCOL)
