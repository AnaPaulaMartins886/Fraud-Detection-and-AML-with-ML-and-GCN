from Functions.Preprocessing import build_graphs
from Functions.Embeddings import get_node2vec_embeddings, run_node2vec_experiment

features_file_path = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_file_path = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edges_file_path = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"

graphs = build_graphs(features_file_path, classes_file_path, edges_file_path)

get_node2vec_embeddings(graphs)

node2vec_results = run_node2vec_experiment()