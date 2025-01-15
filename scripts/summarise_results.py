import sys
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

result_path = sys.argv[1]
finetune_or_pretrain = sys.argv[2]

tests = [
    'prom_prom_300_all',
    'EMP_H3K36me3',
    'mouse_3',
    'EMP_H3K4me2',
    'mouse_4',
    'mouse_1',
    'EMP_H3',
    'EMP_H3K9ac',
    'mouse_2',
    'tf_1',
    'mouse_0',
    'tf_0',
    'EMP_H4ac',
    'prom_prom_core_all',
    'EMP_H3K4me3',
    'prom_prom_300_tata',
    'tf_3',
    'prom_prom_300_notata',
    'prom_prom_core_notata',
    'EMP_H3K14ac',
    'EMP_H4',
    'EMP_H3K4me1',
    'prom_prom_core_tata',
    'tf_2',
    'EMP_H3K79me3',
    'tf_4'
]

if finetune_or_pretrain == "finetune":
    models = [
        'dnabert1_3',
        'dnabert1_6',
        'dnabert2',
        'nt2_500_multi',
        'nt_500_1000g'
    ]
    model_order = ['dnabert1_3', 'dnabert1_6', 'dnabert2', 'nt_500_1000g', 'nt2_500_multi']
    color_mapping = {
        'dnabert1_3': 'red',
        'dnabert1_6': 'green',
        'dnabert2': 'blue',
        'nt_500_1000g': 'orange',
        'nt2_500_multi': 'purple'
    }
else: 
    models = [
        'dnabert2',
        'dnabert2_OG',
        'dnabert9_1',
        'dnabert40_1',
        'dnabert100_20'
    ]
    model_order = ['dnabert2_OG', 'dnabert2', 'dnabert9_1', 'dnabert40_10', 'dnabert100_20']
    color_mapping = {
        'dnabert2_OG': 'red',
        'dnabert2': 'green',
        'dnabert9_1': 'blue',
        'dnabert40_10': 'orange',
        'dnabert100_20': 'purple'
    }
    tests.extend(['Cancer','Cancer2'])

num_tests = len(tests)+2
num_cols = 5
num_rows = math.ceil(num_tests / num_cols)

results = {
    'model': [],
    'test': [],
    'loss': [],
    'accuracy': [],
    'f1': [],
    'matthews_correlation': [],
    'precision': [],
    'recall': [],
    'runtime': [],
    'samples_per_second': [],
    'steps_per_second': [],
    'epoch': []
}

def get_directories(path):
    directories = []
    for item in os.listdir(path):                   # Iterate over the items in the given path
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):                # Check if it is a directory
            directories.append(item)
    return directories

print(f"All of the results are located: {result_path}")

models = get_directories(result_path)

for model in models:
    model_result = os.path.join(result_path, model, "results/")
    print(model_result)
    gue_test_results = get_directories(model_result)

    # There should be a total of 28 tests
    if len(gue_test_results) != num_tests:
        missing_tests = num_tests - len(gue_test_results)
        print(f"ERROR: {model} is missing {missing_tests} test.")
        #exit(1)

    for ind_test in gue_test_results:
        ind_test_results = os.path.join(model_result, ind_test, "eval_results.json")
        ind_test = re.split('_multi_|_3e-5_|_1000g_|1_6_|1_3_', ind_test)[1].rsplit('_seed42')[0]
        with open(ind_test_results, "r") as f:
            ind_result = json.load(f)
            results['model'].append(f'{model}')
            results['test'].append(f'{ind_test}')
            results['loss'].append(ind_result.get('eval_loss'))
            results['accuracy'].append(ind_result.get('eval_accuracy'))
            results['f1'].append(ind_result.get('eval_f1'))
            results['matthews_correlation'].append(ind_result.get('eval_matthews_correlation'))
            results['precision'].append(ind_result.get('eval_precision'))
            results['recall'].append(ind_result.get('eval_recall'))
            results['runtime'].append(ind_result.get('eval_runtime'))
            results['samples_per_second'].append(ind_result.get('eval_samples_per_second'))
            results['steps_per_second'].append(ind_result.get('eval_steps_per_second'))
            results['epoch'].append(ind_result.get('epoch'))
        
# Create a DataFrame from the results dictionary
df = pd.DataFrame(results)

# Display the DataFrame
df.to_csv("final_results_dataframe.csv", index=False)

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), constrained_layout=True)
# Flatten the axes array for easy iteration
axes = axes.flatten()

df['model'] = pd.Categorical(df['model'], categories=models, ordered=True)
for i, test in enumerate(tests):
    # Filter the DataFrame for the current test
    filtered_df = df[df['test'] == test]
    filtered_df.loc[:, 'model'] = pd.Categorical(filtered_df['model'], categories=model_order, ordered=True)

    # Sort the DataFrame by 'matthews_correlation' to get the order for coloring
    #sorted_df = filtered_df.sort_values(by='matthews_correlation', ascending=False)

    # Sort the DataFrame by the 'models' column
    sorted_df = filtered_df.sort_values(by='model')

    # Generate colors based on the sorted values
    #colors = cm.RdYlGn(sorted_df['matthews_correlation'] / sorted_df['matthews_correlation'].max())
    colors = sorted_df['model'].map(color_mapping)

  
    # Plot the matthews_correlation for the current test with colors
    bars = axes[i].bar(sorted_df['model'], sorted_df['matthews_correlation'], color=colors)
    
    # Plot the matthews_correlation for the current test
    #axes[i].bar(filtered_df['model'], filtered_df['matthews_correlation'])
    axes[i].set_title(test)
    #axes[i].set_xlabel('Test')
    axes[i].set_ylim(top=1.0)
    axes[i].set_ylabel('Matthews Correlation')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Save the plot to a file
plt.savefig('matthews_correlation_plot.pdf')

# Optionally, close the plot to free up memory
plt.close()