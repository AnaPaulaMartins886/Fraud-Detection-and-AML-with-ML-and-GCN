from Functions.Preprocessing import build_graphs
from Functions.Embeddings import get_node2vec_embeddings, run_node2vec_experiment, run_deepwalk_experiment, get_deepwalk_embeddings
from Functions.Supervised_Experiments import run_supervised_gcn

from Functions.config import features_file_path, classes_file_path, edges_file_path

graphs = build_graphs(features_file_path, classes_file_path, edges_file_path)

get_node2vec_embeddings(graphs)
get_deepwalk_embeddings(graphs)


gcn_embeddings_results = run_supervised_gcn()

node2vec_results = run_node2vec_experiment()

deepwalk_results = run_deepwalk_experiment()