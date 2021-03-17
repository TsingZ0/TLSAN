# TLSAN
This is the implementation for our published paper: [TLSAN: Time-aware Long- and Short-term Attention Network for Next-item Recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0925231221002605). The full-text is also available here: https://www.researchgate.net/publication/349912702_TLSAN_Time-aware_Long-_and_Short-term_Attention_Network_for_Next-item_Recommendation. Here are the brief introductions to the dataset and the experiment results. 

## Environments
- Python=3.5
- Tensorflow=1.8.0
- Numpy=1.14.2
- Pandas=0.24.1

## Datasets
Amazon exposes the official datasets (http://jmcauley.ucsd.edu/data/amazon/) which have filtered out users and items with less than 5 reviews and removed a large amount of invalid data. Because of above advantages, these datasets are widely utilized by researchers. We also chose Amazon's dataset for experiments. In our experiments, only users, items, interactions, and category information are utilized. We do the preprocessing in the following two steps:
1. Remove the users whose interactions less than 10 and the items which interactions less than 8 to ensure the effectiveness of each user and item.
2. Select the users with more than 4 sessions, and select up to 90 behavior records for the remaining users. This step guarantees the existence of long- and short-term behavior records and all behavior records occurred within recent three months.

### Statistics (after preprocessing)
Datasets | users | items | categories | samples | avg.<br>items/cate | avg.<br>behaviors/item | avg.<br>behaviors/user
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:
Electronics | 39991 | 22048 | 673 | 561100 | 32.8 | 25.4 | 14.0
CDs-Vinyl | 24179 | 27602 | 310 | 470087 | 89.0 | 17.0 | 19.4
Clothing-Shoes | 2010 | 1723 | 226 | 13157 | 7.6 | 7.6 | 6.5
Digital-Music | 1659 | 1583 | 53 | 28852 | 29.9 | 18.2 | 17.4
Office-Products | 1720 | 901 | 170 | 29387 | 5.3 | 32.6 | 17.0
Movies-TV | 35896 | 28589 | 15 | 752676 | 1905.9 | 20.9 | 26.3
Beauty | 3783 | 2658 | 179 | 54225 | 14.8 | 20.4 | 14.3
Home-Kitchen | 11567 | 7722 | 683 | 143088 | 11.3 | 12.3 | 18.5
Video-Games | 5436 | 4295 | 58 | 83748 | 74.1 | 19.5 | 15.4
Toys-and-Games | 2677 | 2474 | 221 | 37515 | 11.2 | 15.2 | 14.0

## Experiment results
Datasets | ATRank | BPR-MF | CNN | CSAN– | LSPM | PACA | Bi-LSTM | SHAN | TLSAN
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:
Electronics | <ins>0.8659</ins> | 0.7457 | 0.8450 | 0.8005 | 0.7333 | 0.8322 | 0.8495 | 0.7542 | **0.9230**
CDs-Vinyl | <ins>0.8999</ins> | 0.7684 | 0.8438 | 0.7943 | 0.8594 | 0.8919 | 0.8969 | 0.7138 | **0.9651**
Clothing-Shoes | 0.6761 | 0.6283 | 0.6712 | 0.5866 | 0.6443 | 0.5313 | 0.7004 | <ins>0.7284</ins> | **0.9363**
Digital-Music | 0.8601 | 0.7896 | 0.8131 | 0.7685 | 0.8270 | <ins>0.9638</ins> | 0.8468 | 0.7794 | **0.9753**
Office-Products | 0.9162 | 0.5610 | 0.8930 | 0.8401 | 0.7889 | 0.8994 | 0.8628 | <ins>0.9576</ins> | **0.9773**
Movies-TV | 0.8662 | 0.7654 | 0.7479 | 0.7958 | 0.8003 | 0.8055 | <ins>0.8743</ins> | 0.7771 | **0.8986**
Beauty | 0.8160 | 0.6846 | 0.7639 | 0.7620 | 0.7748 | <ins>0.9016</ins> | 0.8231 | 0.8953 | **0.9368**
Home-Kitchen | 0.7039 | 0.6352 | 0.7075 | 0.6820 | 0.6672 | 0.8165 | 0.7373 | <ins>0.8230</ins> | **0.8950**
Video-Games | 0.8809 | 0.6609 | 0.8598 | 0.8033 | 0.8449 | 0.8763 | 0.8598 | <ins>0.9216</ins> | **0.9459**
Toys-Games | 0.8139 | 0.6294 | 0.7788 | 0.7157 | 0.7708 | 0.8495 | 0.8012 | <ins>0.8797</ins> | **0.9309**

## How to run the codes
Download raw data and preprocess it with utils:
```
sh 0_download_raw.sh
python3 1_convert_pd.py
python3 2_remap_id.py
```
Build dataset for each model:
```
python3 build_dataset.py
```
Train and evaluate the model:
```
python3 train.py
```
