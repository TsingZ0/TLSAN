#! /bin/bash

cd ../raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz
gzip -d reviews_Office_Products_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz
gzip -d meta_Office_Products.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
gzip -d reviews_Clothing_Shoes_and_Jewelry_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz
gzip -d meta_Clothing_Shoes_and_Jewelry.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
gzip -d reviews_Digital_Music_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz
gzip -d meta_Digital_Music.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
gzip -d reviews_Beauty_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
gzip -d meta_Beauty.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz
gzip -d reviews_Home_and_Kitchen_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz
gzip -d meta_Home_and_Kitchen.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
gzip -d reviews_Video_Games_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz
gzip -d meta_Video_Games.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
gzip -d reviews_Toys_and_Games_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz
gzip -d meta_Toys_and_Games.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
gzip -d reviews_Books_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gzip -d meta_Books.json.gz