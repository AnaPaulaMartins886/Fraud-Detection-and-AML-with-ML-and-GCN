import pandas as pd
import networkx as nx
import numpy as np


def rename_features(features):
    # Define new column names using dictionaries.
    col_names1 = {"0": "id", 1: "time_step"}
    col_names2 = {str(ii + 2): "Local_feature_" + str(ii + 1) for ii in range(93)}
    col_names3 = {str(ii + 95): "Aggregate_feature_" + str(ii + 1) for ii in range(72)}

    # Combine dictionaries and convert keys to integers.
    col_names = {**col_names1, **col_names2, **col_names3}
    col_names = {int(key): value for key, value in col_names.items()}

    # Rename columns using the new column names.
    features = features.rename(columns=col_names)

    return features


def transform_labels_to_int(classes):
    # Define a mapping dictionary for label transformation
    label_mapping = {'1': 1, '2': 0, 'unknown': -1}
    
    # Replace string class labels with integer values using the mapping dictionary
    classes['class'] = classes['class'].replace(label_mapping)
    
    return classes


def get_nodes(features, classes, edges):
    
    all_nodes = set(edges['txId1']).union(edges['txId2']).union(classes['txId']).union(features['id'])
    
    nodes_df = pd.DataFrame(all_nodes, columns=['id']).reset_index()
    
    return nodes_df


def redefine_edges_index(edges, nodes_df):
    
    edges = edges.join(nodes_df.set_index('id'), on='txId1', how='inner') \
                 .join(nodes_df.set_index('id'), on='txId2', how='inner', rsuffix='2') \
                 .drop(columns=['txId1', 'txId2']) \
                 .rename(columns={'index': 'txId1', 'index2': 'txId2'})
    
    return edges


def redefine_classes_index(classes, nodes_df):
    
    classes = classes.join(nodes_df.set_index('id'), on='txId', how='inner') \
                     .drop(columns=['txId']) \
                     .rename(columns={'index': 'txId'})[['txId', 'class']]
    return classes


def redefine_features_index(features, nodes_df):
    
    features = features.join(nodes_df.set_index('id'), on='id', how='inner') \
                       .drop(columns=['id']) \
                       .rename(columns={'index': 'id'})
    
    features = features[['id'] + list(features.columns.drop('id'))]
    
    return features


def redefine_indexes(edges, classes, features, nodes_df) :
    
    edges = redefine_edges_index(edges, nodes_df)
    classes = redefine_classes_index(classes, nodes_df)
    features = redefine_features_index(features, nodes_df)
    
    return edges, classes, features


def get_time_steps(edges, classes, features): # Gets timesteps for classes and edges
    
    # Joining 'features' DataFrame twice with 'edges' DataFrame
    edges = edges.join(
        features[['id', 'time_step']].rename(columns={'id': 'txId1'}).set_index('txId1'),
        on='txId1',
        how='left',
        rsuffix='1'
    ).join(
        features[['id', 'time_step']].rename(columns={'id': 'txId2'}).set_index('txId2'),
        on='txId2',
        how='left',
        rsuffix='2'
    )

    # Selecting desired columns and renaming them
    edges = edges[['txId1', 'txId2', 'time_step']].rename(columns={'txId1': 'source', 'txId2': 'target', 'time_step': 'time_step'})

    # Renaming columns and sorting 'classes' DataFrame
    classes = classes.rename(columns={'txId': 'nid', 'class': 'class'})[['nid', 'class']].sort_values(by='nid')

    # Merging 'classes' DataFrame with 'features' DataFrame based on 'nid' column
    classes = classes.merge(features[['id', 'time_step']].rename(columns={'id': 'nid'}), on='nid', how='left')

    return edges, classes


def merge_classes_and_features(classes, features, only_labeled = False) :

    # Merge classes and features using the 'txId' column.
    classes_features = classes.merge(features.rename(columns={'id':'nid'}).drop(columns=['time_step']),on='nid',how='left')

    # Select only the labeled transactions (i.e. class label is not -1).
    if(only_labeled):
        classes_features = classes_features.loc[(classes_features['class'] != -1)]
    
    return classes_features


def merge_classes_and_edges(classes, edges, only_labeled=False):
    
    # Merge classes and edges on 'source' column
    edges_classes = edges.merge(classes, left_on='source', right_on='nid', how='left')
    edges_classes = edges_classes.rename(columns={'class': 'class1'}).drop(columns=['nid'])

    # Merge classes and edges on 'target' column
    edges_classes = edges_classes.merge(classes, left_on='target', right_on='nid', how='left')
    edges_classes = edges_classes.rename(columns={'class': 'class2'}).drop(columns=['nid'])

    # Select only the labeled transactions (i.e. class labels are not -1)
    if only_labeled:
        edges_classes = edges_classes[(edges_classes['class1'] != -1) & (edges_classes['class2'] != -1)]

    return edges_classes


def read_data_supervised(features_file_path, classes_file_path, edges_file_path):
    print("Reading dataset")
    
    # Read dataset
    features = pd.read_csv(features_file_path, header=None)
    classes = pd.read_csv(classes_file_path)
    edges = pd.read_csv(edges_file_path)
    
    # Rename features dataframe
    features = rename_features(features)
    
    # Transform labels to int
    classes = transform_labels_to_int(classes)
    
    # Get nodes
    nodes_df = get_nodes(features, classes, edges)
    
    # Redefine indexes
    edges, classes, features = redefine_indexes(edges, classes, features, nodes_df)
    
    # Get time steps for classes and edges
    edges, classes = get_time_steps(edges, classes, features)
    
    # Merge classes and features (get the class for the features)
    class_features = merge_classes_and_features(classes, features)
    
    # Merge classes and edges (get the class for the edges)
    edges_classes = merge_classes_and_edges(classes, edges)

    return class_features, edges_classes, edges


def read_data_gcn(features_file_path, classes_file_path, edges_file_path):

    #read dataset
    features = pd.read_csv(features_file_path, header=None)
    classes = pd.read_csv(classes_file_path)
    edges = pd.read_csv(edges_file_path)
    
    #rename features dataframe
    features = rename_features(features)
    
    #transform labels to int
    classes = transform_labels_to_int(classes)
    
    nodes_df = get_nodes(features, classes, edges)
    
    edges, classes, features = redefine_indexes(edges, classes, features, nodes_df)
    
    edges, classes = get_time_steps(edges, classes, features)

    return features, classes, edges


def pre_processing(df, only_labeled=False):
    
    # Order the dataframe by time_step and drop nid column
    df = df.sort_values(by=['time_step', 'nid']).drop(columns=['nid'])

    if only_labeled:
        # Select only labeled transactions
        df = df.loc[df['class'] != -1]

    return df


def get_data(features, classes, get_all=True, only_labeled=False):
    
    # Copy the features DataFrame
    df = features.copy()
    
    # Filter columns based on the 'get_all' parameter
    if not get_all:
        aggregate_columns = df.columns[df.columns.str.contains("Aggregate_feature_")]
        df = df.drop(columns=aggregate_columns)
    
    # Perform pre-processing based on the 'only_labeled' parameter
    df = pre_processing(df, only_labeled)
    
    # Reset the index and remove the 'index' column
    df = df.reset_index(drop=True)
    
    return df


def get_data_gcn(df, get_all=True, only_labeled=False, only_gcn_embedding=False):
    if get_all:
        selected_columns = df.columns
        
    elif only_gcn_embedding:
        selected_columns = df.columns[~df.columns.str.contains("Local_feature_|Aggregate_feature_")]
        
    else:
        selected_columns = df.columns[~df.columns.str.contains("Aggregate_feature_")]
        

    df = df[selected_columns]
    df = pre_processing(df, only_labeled)
        
    return df


def get_data_for_gcn(features_file_path, classes_file_path, edges_file_path) :
    
    (features, classes, edges) = read_data_gcn(features_file_path, classes_file_path, edges_file_path)
    
    class_features = merge_classes_and_features(classes, features)
    print("Nr transactions: ", len(class_features))
    
    edges_classes = merge_classes_and_edges(classes, edges)
    
    print("Nr edges: ", len(edges_classes))

    return class_features, edges_classes


def build_graphs_by_timestep(features, classes, edges, only_labeled=False):
    # Merge classes and features (because we need the node id, class, and timestep)
    classes_features = merge_classes_and_features(classes, features, only_labeled=only_labeled)
    
    if only_labeled:
        edges_classes = merge_classes_and_edges(classes, edges, only_labeled=only_labeled)
    
    # Get unique timesteps
    timesteps = np.sort(classes_features['time_step'].unique())
    
    # Create an empty dictionary to store graphs by timestep
    graphs_by_timestep = {}
    
    # Loop over timesteps and build a graph for each timestep
    for timestep in timesteps:
        print("Building graph for timestep", timestep)
        
        # Filter data for the current timestep
        df = classes_features[classes_features['time_step'] == timestep].copy()
        edges_df = edges[edges['source'].isin(df['nid']) & edges['target'].isin(df['nid'])].copy()
        
        # Reindex nodes
        nid_to_new_index = {nid: i for i, nid in enumerate(df['nid'])}
        df['nid'] = df['nid'].map(nid_to_new_index)
        edges_df['source'] = edges_df['source'].map(nid_to_new_index)
        edges_df['target'] = edges_df['target'].map(nid_to_new_index)
        
        # Reset indexes
        df.reset_index(drop=True, inplace=True)
        edges_df.reset_index(drop=True, inplace=True)
        
        # Create an empty graph
        graph = nx.DiGraph()
        
        # Add nodes
        edge_list = [(row['source'], row['target']) for _, row in edges_df.iterrows()]
        graph.add_edges_from(edge_list)
        
        # Get node attributes
        node_attributes = df.set_index('nid')
        
        # Set the normalized node attributes to the graph
        nx.set_node_attributes(graph, node_attributes.to_dict('index'))
        
        # Add the graph to the dictionary of graphs by timestep
        graphs_by_timestep[timestep] = graph
    
    return graphs_by_timestep