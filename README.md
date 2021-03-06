# TLSAN
This is our implementation for our paper: TLSAN: Time-aware Long- and Short-term Attention Network for Next-item Recommendation

## Environments
- Python=3.5
- Tensorflow=1.8.0
- numpy=1.14.2
- pandas=0.24.1

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
