import numpy as np
import pandas as pd
import random

from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler

from Preprocessing import get_data_for_gcn
from Embeddings import get_gcn_embeddings_test_data, get_gcn_embeddings_train_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

features_file_path = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_file_path = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edges_file_path = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"


class NormalGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, dropout):
        super(NormalGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data, training=False, return_embeddings=False):
        x = self.conv1(data.x, data.edge_index)
        if return_embeddings:
            return x
        x = x.relu()
        x = self.dropout(x) if training else x
        x = self.conv2(x, data.edge_index)
        return x


class SkipGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, dropout):
        super(SkipGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels + num_node_features, num_classes)
        self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, num_classes)))

    def forward(self, data, training=False, return_embeddings=False):
        x = self.conv1(data.x, data.edge_index)
        if return_embeddings:
            return x
        x1 = x.relu()
        x1 = self.dropout(x1) if training else x1
        x2 = torch.cat([data.x, x1], dim=-1)
        x2 = self.conv2(x2, data.edge_index)
        x2 = x2 + torch.matmul(data.x, self.weight)
        return x2


def build_data_loaders(df_edges, df_nodes):
    train_dataset = []
    test_dataset = []

    for i in range(49):
        nodes_df_tmp = df_nodes[df_nodes['time_step'] == i + 1].reset_index()
        nodes_df_tmp['index'] = nodes_df_tmp.index

        df_edge_tmp = df_edges.join(nodes_df_tmp.rename(columns={'nid': 'source'})[['source', 'index']].set_index('source'), on='source', how='inner') \
            .join(nodes_df_tmp.rename(columns={'nid': 'target'})[['target', 'index']].set_index('target'), on='target', how='inner', rsuffix='2') \
            .drop(columns=['source', 'target']) \
            .rename(columns={'index': 'source', 'index2': 'target'})

        # Normalize node features
        node_features = nodes_df_tmp.sort_values(by='index').drop(columns=['index', 'nid', 'class'])
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(node_features)
        x = torch.tensor(normalized_features, dtype=torch.float)

        edge_index = torch.tensor(np.array(df_edge_tmp[['source', 'target']]).T, dtype=torch.long)
        
        y = torch.tensor(np.array(nodes_df_tmp['class']))

        if i + 1 < 35:
            data = Data(x=x, edge_index=edge_index, y=y)
            train_dataset.append(data)
        else:
            data = Data(x=x, edge_index=edge_index, y=y)
            test_dataset.append(data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_nodes = len(df_nodes)
    print("Number of nodes:", num_nodes)

    num_train_batches = len(train_loader)
    print("Number of batches in train_loader:", num_train_batches)

    return train_loader, test_loader


def load_data() :
    
    normalized_class_features, edges_classes = get_data_for_gcn(features_file_path, classes_file_path, edges_file_path)
    
    train_data, test_data = build_data_loaders(edges_classes, normalized_class_features)
    
    return train_data, test_data


def split_train_test_data(df_edges, df_nodes) :
    
    timestep = 35
    
    # Split nodes into training and test sets based on time step
    train_nodes = df_nodes.loc[df_nodes['time_step'] < timestep]
    test_nodes = df_nodes.loc[df_nodes['time_step'] >= timestep]
    
    train_nodes = train_nodes.reset_index()
    train_nodes['index'] = train_nodes.index
    
    test_nodes = test_nodes.reset_index()
    test_nodes['index'] = test_nodes.index


def instantiate_models_GCN(in_features, hidden_features, out_features, dropout):
    
    # Initialize GCN models
    normal_gcn_model = NormalGCN(in_features, hidden_features, out_features, dropout)
    skip_gcn_model = SkipGCN(in_features, hidden_features, out_features, dropout)


    # Return a list of tuples, each containing a model name and the corresponding model instance.
    models = [
        ('Normal GCN', normal_gcn_model),
        ('Skip GCN', skip_gcn_model)
    ]
    
    return models


def define_training_parameters(model, device, learning_rate, weight_decay) :
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3,0.7]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return criterion, optimizer


def train(model, train_data_loader, device, epochs, learning_rate, weight_decay):
    
    criterion, optimizer = define_training_parameters(model, device, learning_rate, weight_decay)
    
    model.to(device)

    # Train model
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for data in train_data_loader:
            #print(data.x.shape)
            #print(data.edge_index.shape)
            #print(data.y.shape)
            model.train()
            optimizer.zero_grad()
            out = model(data)
            node_embeddings = model(data, return_embeddings=True)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_data_loader):.4f}")

    return model, node_embeddings


def evaluate(model, test_data_loader, device):
    
    model.eval()

    all_probabilities = []
    all_node_embeddings = []
    all_y_true = []

    with torch.no_grad():
        for data in test_data_loader:
            data = data.to(device)
            output = model(data)
            node_embeddings = model(data, return_embeddings=True)
            probabilities = torch.softmax(output, dim=1)

            all_probabilities.append(probabilities)
            all_node_embeddings.append(node_embeddings)
            all_y_true.append(data.y)  # Append the ground truth labels

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_node_embeddings = torch.cat(all_node_embeddings, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)


    return all_probabilities, all_node_embeddings, all_y_true


def run_model_GCN(model, train_data_loader, test_data_loader, num_epochs, lr):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    model, node_embeddings_train = train(model, train_data_loader, device, epochs=num_epochs, learning_rate=lr, weight_decay=0.01) # train the model for a given number of epochs and learning rate
    
    # Evalaluate the model
    probabilities, node_embeddings_test, y_true = evaluate(model, test_data_loader, device)

    return probabilities, node_embeddings_train, node_embeddings_test, y_true


def calculate_scores_GCN(y_preds, y_true):
    
    # Initialize metrics variables
    f1_scores = []
    recall_scores = []
    precision_scores = []
    micro_f1_scores = []
    best_thresholds = []
    predicted_labels = []

    for y_pred in y_preds:
        y_test = y_true

        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])
    
        # Calculate F1 scores for each threshold
        score = [f1_score(y_test, y_pred[:, 1] > threshold, pos_label=1) for threshold in thresholds]
        micro_scores = [f1_score(y_test, y_pred[:, 1] > threshold, pos_label=1, average='micro') for threshold in thresholds]

        # Find the threshold that gives the highest F1 score
        max_f1_index = np.argmax(score)
        max_f1_threshold = thresholds[max_f1_index]
        max_f1_score = score[max_f1_index]
        
        # Find the precision and recall at the highest F1 score
        best_precision = precision[max_f1_index + 1]
        best_recall = recall[max_f1_index + 1]
        
        # Find the micro-averaged F1 score at the highest F1 score threshold
        best_micro_f1_score = micro_scores[max_f1_index]
        
        # Append scores to lists
        f1_scores.append(max_f1_score)
        recall_scores.append(best_recall)
        precision_scores.append(best_precision)
        micro_f1_scores.append(best_micro_f1_score)
        best_thresholds.append(max_f1_threshold)
        
        # Extract the predicted labels using the threshold
        predicted_label = (y_pred[:, 1] > max_f1_threshold).to(torch.int)
        predicted_labels.append(predicted_label)
    
    # Calculate average metrics across all runs
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_micro_f1_score = np.mean(micro_f1_scores)
    
    print("AVG F1 Score ---> ",  avg_f1)
    
    return avg_f1, avg_precision, avg_recall, avg_micro_f1_score, best_thresholds, predicted_labels


def run_GCN() -> pd.DataFrame:

    # Get Data
    train_data, test_data = load_data()

    
    # Define the hyperparameters
    in_channels = 166
    hidden_channels = 100
    out_channels = 2
    lr = 0.001
    num_epochs = 1000
    num_runs = 1
    dropout = 0.5
     

    results = [] # Initialize results list
    normal_gcn_preds = [] # list to hold predictions for normal GCN models
    skip_gcn_preds = [] # list to hold predictions for skip GCN models
    
    model_results = {} # Initialize a dictionary to hold the evaluation results for each model
    
    
    np.random.seed(8347658)
    
    for a in range(num_runs):
        
        print("Run", a + 1)
        
        # Generate a random seed for this run
        seed = random.randint(0, 10000)
        
        # Set the random seed for Torch
        torch.manual_seed(seed)
        
        
        models = instantiate_models_GCN(in_channels, hidden_channels, out_channels, dropout)
    
        # Loop over each model
        for name, model in models:
            
            print("Starting Running", name)
            
            # Run the model multiple times and get predictions
            probabilities, node_embeddings_train, node_embeddings_test, y_true = run_model_GCN(model, train_data, test_data, num_epochs, lr)
            
            if name == 'Normal GCN':
                normal_gcn_preds.append(probabilities)
                
                # Calculate scores for each model
                normal_gcn_scores = calculate_scores_GCN(normal_gcn_preds, y_true)
                
                
            elif name == 'Skip GCN':
                skip_gcn_preds.append(probabilities)

                # Calculate scores for each model
                skip_gcn_scores = calculate_scores_GCN(skip_gcn_preds, y_true)
                
                predicted_labels  = normal_gcn_scores[5]
                get_gcn_embeddings_test_data(model, node_embeddings_test, predicted_labels)
                get_gcn_embeddings_train_data(model, node_embeddings_train)


                
        
    # Add the evaluation results for each model to the dictionary
    model_results['Normal GCN'] = {
        'Precision': normal_gcn_scores[1].round(3),
        'Recall': normal_gcn_scores[2].round(3),
        'F1 Score': normal_gcn_scores[0].round(3),
        'Micro AVG F1': normal_gcn_scores[3].round(3)
    }
    
    model_results['Skip GCN'] = {
        'Precision': skip_gcn_scores[1].round(3),
        'Recall': skip_gcn_scores[2].round(3),
        'F1 Score': skip_gcn_scores[0].round(3),
        'Micro AVG F1': skip_gcn_scores[3].round(3)
    }
        
    # Append results for each model to the list
    for name, results_dict in model_results.items():
        results.append({
            'Model Name': name,
            **results_dict
        })

        
    return pd.DataFrame(results)


