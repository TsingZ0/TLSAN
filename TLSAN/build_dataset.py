import random
import pickle
import numpy as np
import pandas as pd
import copy

max_length = 90
random.seed(1234)

with open('../Data/Digital_Music.pkl', 'rb') as f:
    reviews_df, meta_df = pickle.load(f)
    item_cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)
    
# generate current time positions
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

def proc_time_emb(hist_t, cur_t):
    hist_t = [cur_t - i + 1 for i in hist_t]
    hist_t = [1/np.sum(i >= gap) for i in hist_t]
    return hist_t

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()
    tim_list = hist['unixReviewTime'].tolist()
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg
    neg_list = [gen_neg() for i in range(len(pos_list))]

    length = len(pos_list)
    valid_length = min(length, max_length)
    i = 0
    tim_list_session = list(set(tim_list))
    tim_list_session.sort()
    pre_session = []
    pre_time = []
    pre_cates = []
    for t in tim_list_session:
        count = tim_list.count(t)
        new_session = pos_list[i:i+count]
        new_time = tim_list[i:i+count]
        new_cates = [meta_df[meta_df['asin'] == item]['categories'].tolist()[0] for item in new_session]

        if t == tim_list_session[0]:
            pre_session.extend(new_session)
            pre_time.extend(new_time)
            pre_cates.extend(new_cates)
        else:
            now_cate = pd.value_counts(pre_cates).index[0]
            if i+count < valid_length-1:
                pre_time_emb = proc_time_emb(pre_time, tim_list[i])
                pre_session_copy = copy.deepcopy(pre_session)
                train_set.append((reviewerID, pre_session_copy, new_session, pre_time_emb, pos_list[i+count], 1, now_cate))
                train_set.append((reviewerID, pre_session_copy, new_session, pre_time_emb, neg_list[i+count], 0, now_cate))
                pre_session.extend(new_session)
                pre_time.extend(new_time)
                pre_cates.extend(new_cates)
            else:
                pos_item = pos_list[i]
                if count > 1:
                    pos_item = random.choice(new_session)
                    new_session.remove(pos_item)
                neg_index = pos_list.index(pos_item)
                pos_neg = (pos_item, neg_list[neg_index])
                pre_time_emb = proc_time_emb(pre_time, t)
                test_set.append((reviewerID, pre_session, new_session, pre_time_emb, pos_neg, now_cate))
                break
        i += count

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_cate_list, f, pickle.HIGHEST_PROTOCOL)
