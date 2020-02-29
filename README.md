# TLSAN
This is our implementation for our paper: TLSAN: Time-aware Long- and Short-term Attention Network for Next-item Recommendation

## Environments
- Python >= 3.5
- Tensorflow 1.8.0
- numpy
- pandas

## Dataset
Amazon also exposes the official datasets\footnote{http://jmcauley.ucsd.edu/data/amazon/} which have filtered out users and items with less than 5 reviews and removed a large amount of invalid data. Because of above advantages, these datasets are widely utilized by researchers. We also chose Amazon's dataset for experiments. In the following experiments, only users, items, interactions, and category information are utilized. Then we do the preprocessing in the following two steps:
1. Remove the users whose interactions less than 10 and the items which interactions less than 8 to ensure the effectiveness of each user and item.
2. Select the users with more than 4 sessions, and select up to 90 behavior records for the remaining users. This step guarantees the existence of long- and short-term behavior records and all behavior records occurred within recent three months.

## How to run the codes
Build dataset:
```
python3 build_dataset.py
```
Train and evaluate the model:
```
python3 train.py
```
