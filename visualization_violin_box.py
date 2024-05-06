import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to create scatter plots
def scatter_plot(nrow, ncol, exp, title):
    # Create subplots based on the specified number of rows and columns
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    
    # Iterate through each subplot
    for i, ax in enumerate(axes.flatten()):
        # Read the data from the CSV file based on the experiment number
        df = pd.read_csv('exp_result/experiment_' + str(exp[i]) + '_questions_with_ratings.csv')
        # Select relevant columns from the DataFrame
        df = df[['ID', 'relevance_score', 'groundedness_score', 
                 'dontknowness_score', 'coherence_score']]
        # Reshape the DataFrame to long format for plotting
        df_long = pd.melt(df, id_vars=['ID'], 
                          value_vars=['relevance_score', 'groundedness_score', 
                                      'dontknowness_score', 'coherence_score'])
        
        # Jitter the y-axis values for better visualization
        y_jittered = [y + np.random.uniform(-0.2, 0.2) for y in df_long['value']]
        data = {'Rating': df_long['variable'], 'Score': y_jittered}
        # Create scatter plot
        sns.stripplot(x='Rating', y='Score', data=data, jitter=True, alpha=0.3, ax=ax)
        ax.set_title(title[i])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['relevance', 'groundedness', 'dontknowness', 'coherence'])
    
    plt.tight_layout()
    plt.show()

# Function to create violin plots
def violin_plot(nrow, ncol, exp, title):
    # Create subplots based on the specified number of rows and columns
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 5*nrow))
    
    # Iterate through each subplot
    for i, ax in enumerate(axes.flatten()):
        # Read the data from the CSV file based on the experiment number
        df = pd.read_csv('exp_result/experiment_' + str(exp[i]) + '_questions_with_ratings.csv')
        # Select relevant columns from the DataFrame
        df = df[['ID', 'relevance_score', 'coherence_score', 'groundedness_score', 'gpt_similarity']]
        # Calculate overall score
        df['overall_score'] = df[['relevance_score', 'coherence_score', 
                                  'groundedness_score', 'gpt_similarity']].mean(axis=1)
        # Rename columns for better visualization
        df.rename(columns={'relevance_score': 'Relevance', 'coherence_score': 'Coherence', 
                           'groundedness_score': 'Groundedness', 'gpt_similarity': 'GPT Similarity', 
                           'overall_score': 'Overall'}, inplace=True)
        
        # Reshape the DataFrame to long format for plotting
        df_long = pd.melt(df, id_vars=['ID'], 
                          value_vars=['Relevance', 'Coherence', 'Groundedness', 
                                      'GPT Similarity', 'Overall'])

        # Create violin plot
        sns.violinplot(x='variable', y='value', data=df_long, ax=ax)
        ax.set_title(title[i])
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['Relevance', 'Coherence', 'Groundedness', 'GPT Similarity', 'Overall'])
        ax.set_xlabel('')
        ax.set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('chunking.png')  # Save the plot as an image
    plt.show()

# Create a DataFrame to store all data
df_all = pd.DataFrame(columns=['ID', 'relevance_score', 'coherence_score', 'groundedness_score', 'gpt_similarity', 
                               'knowness_score', 'overall_score', 'exp'])
# Iterate through each experiment
for i in range(50):
    try:
        # Read the data from the CSV file based on the experiment number
        df = pd.read_csv('exp_result/experiment_' + str(i) + '_questions_with_ratings.csv', 
                         usecols=['ID', 'relevance_score', 'coherence_score', 
                                  'groundedness_score', 'gpt_similarity', 'dontknowness_score'])
        # Drop rows with missing values
        df.dropna(inplace=True)
        # Calculate overall score and knowness score
        df['overall_score'] = df[['relevance_score', 'coherence_score', 'groundedness_score', 'gpt_similarity']].mean(axis=1)
        df['knowness_score'] = 6 - df['dontknowness_score']
        df['knowness_score'] = df['knowness_score'].astype(int)
        df['exp'] = 'exp' + str(i)

        # Concatenate the DataFrame with the main DataFrame
        df_all = pd.concat([df_all, df], join="inner", ignore_index=True)
    
    except FileNotFoundError:
        continue

# Create a boxplot
plt.figure(figsize=(8, 6))
boxes = plt.boxplot([df_all[df_all['knowness_score'] == score]['overall_score'] for score in range(1, 6)],
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='#1f77b4', markeredgecolor='#1f77b4', alpha=0.3))
colors = plt.cm.Blues(np.linspace(0.2, 0.8, 5))
for box, color in zip(boxes['boxes'], colors):
    box.set(facecolor=color, edgecolor='black')

# Add labels and title to the plot
plt.title('Knowness Score vs Overall Score')
plt.xlabel('Knowness Score')
plt.ylabel('Overall Score')
plt.savefig('knowness.png')  # Save the plot as an image
plt.show()