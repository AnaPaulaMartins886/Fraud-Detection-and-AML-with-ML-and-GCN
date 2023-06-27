import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import MinMaxScaler

from Preprocessing import read_data_supervised, get_data, get_data_gcn, build_graphs_by_timestep
from Visualizations import plot_f1_by_timestep, draw_plot

features_file_path = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_file_path = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edges_file_path = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"


def split_data(data):
    train_data = data.loc[data['time_step'] < 35]
    test_data = data.loc[data['time_step'] >= 35]
    
    X_train = train_data.drop(columns=['class'])
    y_train = train_data['class']
    
    X_test = test_data.drop(columns=['class'])
    y_test = test_data['class']
    
    return X_train, y_train, X_test, y_test


def define_models():
    # Initialize the models to test
    models = [
        ('Random Forest', RandomForestClassifier(n_estimators=50, max_features=50)),
        ('Logistic Regression', LogisticRegression(max_iter=2000)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(50,), solver='adam', max_iter=500, learning_rate_init=0.001))
    ]

    return models


def run_model(model, X_train, y_train, X_test, num_runs):
    y_preds = []
    np.random.seed(8347658)  # Set random seed for reproducibility
    
    for _ in range(num_runs):
        random_state = np.random.randint(0, 10000)  # Generate random state for each run
        model.set_params(random_state=random_state)  # Set the random state parameter
        
        model.fit(X_train, y_train)  # Fit the model to the training data
        y_pred = model.predict_proba(X_test)  # Predict probabilities on the test data
        y_preds.append(y_pred)  # Add predictions to the list of predictions for each run
            
    return y_preds  # Return the list of predictions for each run


def calculate_scores(y_preds, y_test, name):
    f1_scores = []
    recall_scores = []
    precision_scores = []
    micro_f1_scores = []
    best_thresholds = []

    for y_pred in y_preds:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])

        scores = [f1_score(y_test, y_pred[:, 1] > threshold, pos_label=1) for threshold in thresholds]
        micro_scores = [f1_score(y_test, y_pred[:, 1] > threshold, pos_label=1, average='micro') for threshold in thresholds]

        max_f1_index = np.argmax(scores)
        max_f1_threshold = thresholds[max_f1_index]
        max_f1_score = scores[max_f1_index]

        best_precision = precision[max_f1_index + 1]
        best_recall = recall[max_f1_index + 1]
        best_micro_f1_score = micro_scores[max_f1_index]

        f1_scores.append(max_f1_score)
        recall_scores.append(best_recall)
        precision_scores.append(best_precision)
        micro_f1_scores.append(best_micro_f1_score)
        best_thresholds.append(max_f1_threshold)
        
        draw_plot(thresholds, precision, recall, scores, max_f1_threshold, max_f1_score, name)

    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_micro_f1_score = np.mean(micro_f1_scores)

    print("Average F1 Score:", avg_f1)

    return avg_f1, avg_precision, avg_recall, avg_micro_f1_score, best_thresholds


def f1_per_timestep(X_test, y_test, y_preds, thresholds):

    last_train_time_step = min(X_test['time_step']) - 1
    last_time_step = max(X_test['time_step'])
    scores_by_ts = []
    i=0

    #itera as predictions de cada modelo ao longo das 5 iteracoes
    for y_pred in y_preds:
        
        model_scores = []
        all_model_scores = []
      
        for time_step in range(last_train_time_step + 1, last_time_step + 1):

            id = np.flatnonzero(X_test['time_step'] == time_step)
            y_true = y_test.iloc[id]

            y__illicit_score = y_pred[:, 1]
            y_pred_ts = [y__illicit_score[i] for i in id]

            
            score = f1_score(y_true.astype('int'), y_pred_ts > thresholds[i], pos_label = 1) 

            model_scores.append(score)

        scores_by_ts.append(model_scores) #guarda o f1-score de cada uma das previsoes do modelo ao longo das 5 iteracoes
        
        i+=1
    
    avg_f1 = np.array([np.mean([f1_scores[i] for f1_scores in scores_by_ts]) for i in range(15)]) #calcula a media de f1 scores para todos os timesteps para cada um dos modelos

    return avg_f1


def calculate_metrics_by_timestep(name, x_test, y_test, y_preds, f1_ts_dict, thresholds):
    
    #metrics by timestep
    f1_ts = f1_per_timestep(x_test, y_test, y_preds, thresholds)
    f1_ts_dict[name] = f1_ts
        
    return f1_ts_dict
        

def run_classifiers(df, experiment_type, num_runs=5):
    
    (X_train, y_train, X_test, y_test) = split_data(df)
    
    # Initialize Classification Models
    models = define_models()
    
    # Initialize results list
    results = []
    f1_ts_dict = {}
    
    # create an instance of the MinMaxScaler class
    scaler = MinMaxScaler()
        
    # Loop over each model
    for name, model in models:
        print("Running", name)
        
        # Normalize the data if the model is MLP
        if name == 'MLP':

            # Exclude 'time_step' column from normalization
            columns_to_normalize = [column for column in df.columns if column != 'time_step' and column != 'class']
            
            # Fit the scaler on the training data for selected columns and transform it
            X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
            
            # Transform the test data for selected columns using the fitted scaler
            X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
            
            # Restore the original 'time_step' column in X_test
            X_test['time_step'] = df.loc[X_test.index, 'time_step']

        
        # Run the model multiple times and get predictions
        y_preds = run_model(model, X_train, y_train, X_test, num_runs)
        
        # Calculate scores for each model
        (avg_f1, avg_precision, avg_recall, avg_micro_f1_score, thresholds) = calculate_scores(y_preds, y_test, name)
        
        # Append results to list
        results.append({
            'Model Name': name,
            'Precision': avg_precision.round(3),
            'Recall': avg_recall.round(3),
            'F1 Score': avg_f1.round(3),
            'Micro AVG F1': avg_micro_f1_score.round(3),
            'Type' : experiment_type
        })
        
        # Calculate F1 Score by Timestep
        f1_ts_dict = calculate_metrics_by_timestep(name, X_test, y_test, y_preds, f1_ts_dict, thresholds)
    
    # Draw Plot F1 Score by Timestep
    plot_f1_by_timestep(f1_ts_dict, experiment_type)
    
    print()
    print()
    print()
    
    print()
    # Return results as a pandas DataFrame
    return pd.DataFrame(results)


def get_datasets_supervised_baseline(classes, features):
    
    # Get All Features
    df_all = get_data(features, classes, True, True)
    
    # Get Local Features
    df_local = get_data(features, classes, False, True)

    # Assign names to each dataset
    datasets = {
        'AF': df_all,
        'LF': df_local
    }

    return datasets


def get_datasets_supervised_gcn(gcn_embeddings):
    
    # Get All Features
    df_all = get_data_gcn(gcn_embeddings, True, True, False)
    
    # Get Local Features
    df_local = get_data_gcn(gcn_embeddings, False, True, False)
    
    #Get Only Embeddings
    df_embeddings = get_data_gcn(gcn_embeddings, False, True, True)

    # Assign names to each dataset
    datasets = {
        'AF + GCN': df_all,
        'LF + GCN': df_local,
        'GCN': df_embeddings
    }

    return datasets


def run_supervised_baseline() :
    
    # Read data
    features, classes, edges = read_data_supervised(features_file_path, classes_file_path, edges_file_path)

    print("Split Datasets")
    datasets = get_datasets_supervised_baseline(classes, features)
    
    # Run experiments on each dataset
    results = []
    
    for dataset_name, dataset in datasets.items():
        dataset_results = run_classifiers(dataset, dataset_name, num_runs = 1)
        results.append(dataset_results)

        
    # Combine results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)
    
    return results_df   


def run_supervised_gcn() :
    
    # Read data
    gcn_embeddings_train = pd.read_csv("Embeddings/gcn_embeddings_training.csv")
    gcn_embeddings_test = pd.read_csv("Embeddings/gcn_embeddings_test.csv")
    
    gcn_embeddings_test = gcn_embeddings_test.drop(columns=['predicted_label'])
    
    # concatenate the two dataframes vertically
    gcn_embeddings = pd.concat([gcn_embeddings_train, gcn_embeddings_test], axis=0)

    # reset the index
    gcn_embeddings.reset_index(inplace=True, drop=True)
    

    
    print("Split Datasets")
    datasets = get_datasets_supervised_gcn(gcn_embeddings)
    
    # Run experiments on each dataset
    results = []
    
    for dataset_name, dataset in datasets.items():
        dataset_results = run_classifiers(dataset, dataset_name, 5)
        results.append(dataset_results)

    # Combine results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)
    
    return results_df


def build_graphs(features_file_path, classes_file_path, edges_file_path) :
    
    print("------------------------------------------GRAPHS-----------------------------------------------------")
    print()
    
    # Read data and build graph
    print("Reading Data")
    features, classes, edges = read_data_supervised(features_file_path, classes_file_path, edges_file_path)
    
    print("Building graph")
    graphs = build_graphs_by_timestep(features, classes, edges, only_labeled = True)
    
    print() 
    
    return graphs