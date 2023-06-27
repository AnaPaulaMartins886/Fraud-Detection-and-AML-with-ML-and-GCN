# Fraud-Detection-and-AML-with-ML-and-GCN

## Overview
This repository includes the necessary code to replicate the results presented in the dissertation titled "Fraud Detection and Anti-Money Laundering applying Machine Learning Techniques in Cryptocurrencies Transactional Graphs". The provided code base covers both the Supervised baseline, the Node Embeddings experiments, and the Graph Convolutional Network experiments.

## Requirements
Please ensure that you have the following main requirements to proceed with the task:

- Python 3.8.10 or a compatible version.
- networkx
- sklearn
- matplotlib
- numpy
- pandas
- torch
- karateclub

Once you have installed the necessary packages, please download the Elliptic Bitcoin dataset from the following link: https://www.kaggle.com/ellipticco/elliptic-data-set. 
After downloading the dataset, save the three .csv files (elliptic_txs_features.csv, elliptic_txs_classes.csv, and elliptic_txs_edgelist.csv) in the elliptic_bitcoin_dataset/ directory.

## Experiments
To reproduce the results, you can find all the necessary Python scripts in the src/Experiments and src/Functions directories. To run a specific experiment, navigate to the project's root folder in the terminal and execute the corresponding Python script:

For the supervised methods: src/Experiments/Supervised_Baseline.py
For the Node Embeddings: src/Experiments/Embeddings.py
For the GCN: src/Experiments/GCN.py

Author
Ana Martins: anapfmartins333@outlook.pt
