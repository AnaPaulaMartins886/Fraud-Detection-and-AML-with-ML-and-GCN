import pandas as pd
import random
import networkx as nx

from karateclub import DeepWalk
from karateclub import Node2Vec

from Preprocessing import get_data_for_gcn
from GCN import split_train_test_data

def get_gcn_embeddings_test_data(model, embeddings, predicted_labels):
    
    embeddings = embeddings.detach().numpy()
    num_embeddings = embeddings.shape[1]
    col_names = [f"GCN_Embedding_{i}" for i in range(num_embeddings)]
    embeddings = pd.DataFrame(embeddings, columns=col_names)
    
    embeddings['predicted_label'] = predicted_labels[-1]
    
    features, edges = get_data_for_gcn()
    _, test_nodes, _, _ = split_train_test_data(edges, features)
    
    embeddings = test_nodes[['class']].merge(embeddings, left_index=True, right_index=True)



    embeddings.to_csv("Embeddings/gcn_embeddings_test.csv", index=False)


def get_gcn_embeddings_train_data(model, embeddings):
    
    embeddings = embeddings.detach().numpy()
    num_embeddings = embeddings.shape[1]
    col_names = [f"GCN_Embedding_{i}" for i in range(num_embeddings)]
    embeddings = pd.DataFrame(embeddings, columns=col_names)


    embeddings.to_csv("Embeddings/gcn_embeddings_training.csv", index=False)


def create_embedding_dataframe(node_ids, node_embeddings, graph):
    
    print("Getting Embeddings")
    # Get the node attributes as a DataFrame
    node_attrs = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')

    # Combine the node embeddings with the attributes
    df_embeddings = pd.concat([pd.DataFrame(node_embeddings), node_attrs], axis=1)

    # Add a column for the node IDs
    df_embeddings.insert(0, "node_id", node_ids)

    # Rename the columns
    df_embeddings = df_embeddings.rename(columns={i: "Embedding_" + str(i) for i in range(node_embeddings.shape[1])})

    return df_embeddings


def run_node2vec(graphs):
    # Randomly assign parameters
    dimensions = 100
    p = random.uniform(0.5, 2)
    q = random.uniform(0.1, 1)
    walk_length = random.randint(10, 200)
    walk_number = random.randint(20, 80)
    epochs = 1000
    
    # Print the randomly assigned parameters
    print("Randomly assigned parameters:")
    print("Dimensions:", dimensions)
    print("p:", p)
    print("q:", q)
    print("Walk Length:", walk_length)
    print("Walk Number:", walk_number)
    print("epochs: ", epochs)
 
    print()
    
    # Initiate Node2Vec Model
    model = Node2Vec(dimensions=dimensions, p=p, q=q, walk_length=walk_length, walk_number=walk_number, epochs=epochs)

    # Get embeddings for each graph
    all_embeddings = []
    
    for timestep, graph in graphs.items():
        print(f"Running Node2Vec on timestep {timestep}/{len(graphs)}")
        graph = nx.convert_node_labels_to_integers(graph, first_label=0)

        model.fit(graph)
        node_ids = list(graph.nodes())
        node_embeddings = model.get_embedding()
        embeddings = create_embedding_dataframe(node_ids, node_embeddings, graph)
        all_embeddings.append(embeddings)

    # Concatenate embeddings for all timesteps
    df_embeddings = pd.concat(all_embeddings)

    return df_embeddings


def get_node2vec_embeddings(graphs):
    
    print("------------------------------------------NODE2VEC-----------------------------------------------------")
    print()
    
    # Get embeddings and split into different datasets
    print("Running Node2Vec")
    embeddings = run_node2vec(graphs)
    print("Running Node2Vec Complete")    
    
    # Save the DataFrame as a CSV file
    embeddings.to_csv("Embeddings/node2vec_embeddings3.csv", index=False)

    return embeddings


def pre_processing_embeddings(df, only_labeled = False) :
    
    df = df.rename(columns={'node_id': 'txId'})
    df = df.sort_values(by = ['time_step', 'txId']) # Order by time step and txId
    df = df.drop(columns=['txId']) # Drop txId
    
    if(only_labeled) :
        df = df.loc[(df['class'] != -1)] #Select only labeled transactions

    return df 


def get_data_embeddings(df, get_all=True, only_labeled = False, only_embedding=False):
    
    if get_all:
        selected_columns = df.columns
        
    elif only_embedding:
        selected_columns = df.columns[~df.columns.str.contains("Local_feature_|Aggregate_feature_")]
        
    else:
        selected_columns = df.columns[~df.columns.str.contains("Aggregate_feature_")]
        
    df = df[selected_columns]
    
    df = pre_processing_embeddings(df, only_labeled = only_labeled)
  
    return df


def get_datasets_embeddings(embeddings, only_labeled = False):

    # Get All Features plus Embeddings
    df_all_embeddings = get_data_embeddings(embeddings, get_all=True, only_labeled = True)

    # Get Local Features plus Embeddings
    df_local_embeddings = get_data_embeddings(embeddings, get_all=False, only_labeled = True)
    
    df_embeddings = get_data_embeddings(embeddings, False, True, True)
    

    # Assign names to each dataset
    datasets = {
        'AF + N2V': df_all_embeddings,
        'LF + N2V': df_local_embeddings,
        'N2V': df_embeddings
    }

    return datasets


def split_data(data):

    trein = data.loc[(data['time_step'] < 35)]
    test = data.loc[(data['time_step'] >= 35)]
    
    X_train = trein.drop(columns=['class'])
    y_train = trein['class']
    
    x_test = test.drop(columns=['class'])
    y_test = test['class']

    
    return (X_train, y_train, x_test, y_test)


def run_node2vec_experiment(num_runs=5):
    
    embeddings = pd.read_csv("Embeddings/node2vec_embeddings3.csv")
    
    print()
    print("------------------------------------------ML MODELS-----------------------------------------------------")
    print()
    
    print("Split Datasets")
    datasets = get_datasets_embeddings(embeddings, only_labeled = True)

    # Run experiments on each dataset
    results = []
    for dataset_name, dataset in datasets.items():
        dataset_results = run_classifiers(dataset, dataset_name, num_runs)
        results.append(dataset_results)

    # Combine results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)
    
    return results_df