import matplotlib.pyplot as plt

from Preprocessing import read_data_supervised

features_file_path = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
classes_file_path = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edges_file_path = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"


def plot_f1_by_timestep(model_metric_dict, experiment_type, last_train_time_step=34,last_time_step=49, fontsize=23, labelsize=18, figsize=(20, 10),
                                  linestyle=['solid', "dotted", 'dashed'], linecolor=["green", "orange", "red"],
                                  barcolor='lightgrey', baralpha=0.3, linewidth=1.5):

    
    (class_features, edges_classes, edges) = read_data_supervised(features_file_path, classes_file_path, edges_file_path)
    data = class_features
    
    occ = data.groupby(['time_step', 'class']).size().to_frame(name='occurences').reset_index()
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]

    timesteps = illicit_per_timestep['time_step'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    i = 0
    
    # Create a new figure for the plots generated by calculate_scores
    plt.figure()
    
    for key, values in model_metric_dict.items():
        
        key = key.lower()
        ax1.plot(timesteps, values, label=key, linestyle=linestyle[i], color=linecolor[i], linewidth=linewidth)

        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha, label='nr illicit transactions')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Time Step', fontsize=fontsize)
    ax1.set_ylabel('Illicit F1', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step+1,last_time_step+1))
    ax1.set_yticks([0,0.25,0.5,0.75,1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Num. illicit transactions', fontsize=fontsize)
    
    plt.title(experiment_type)
    
    # Display the plot
    plt.show()

    return fig


def draw_plot(thresholds, precision, recall, scores, max_f1_threshold, max_f1_score, name):        
    # Create a new figure
    plt.figure()
    
    plt.plot(thresholds, precision[:-1], label='precision')
    plt.plot(thresholds, recall[:-1], label='recall')
    plt.plot(thresholds, scores, label='F1-score')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    
    plt.axvline(x=max_f1_threshold, color='black', linestyle='--')
    plt.text(max_f1_threshold + 0.01, max_f1_score - 0.1, f'Max F1-score = {max_f1_score:.3f}', rotation=90)
    
    title = f"{name}"
    
    plt.title(title)
    plt.legend()
    plt.show()
    
    plt.close()  # Close the plot windo