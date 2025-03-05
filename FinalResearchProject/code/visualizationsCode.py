"""
    This file outlines the code which is contained in the Jupyter notebook for creating various visualizations
    and plots used throughout the project (created in the Jupiter Notebook and explained in the PDF article).
"""

################################################################################################################

"""
    The following function will be called throughout this notebook several times (for each of the 4 datasets), in order to plot an evaluation metrics comparison
    between the performance of the model according to the original pipeline, and the performance of the model after applying:
    (i) each of the 3 sub-algorithms; (ii) each of the 2 baseline algorithms.
"""

def plot_evaluation_metrics_comparison(evaluation_metrics, original_values, *steps_values, step_labels=None, colors=None):
    # In each "step", different algorithm out of the 6 (3 sub-algorithms, 2 baseline algorithm and 1 original pipeline) is applied, and the change in the evaluation metrics is calculated (in %).
    num_steps = len(steps_values)
    fig_width = 10 + (num_steps - 1) * 1.5
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    barWidth = 0.8 / (num_steps + 1)
    r1 = np.arange(len(evaluation_metrics))

    # I also add a line representing the original pipeline (0% change) to make the comparison clearer:
    ax.axhline(0, color='black', linewidth=2, linestyle='--', label='Original Pipeline (0% change)')

    # Calculating the change in % for each algorithm:
    percentage_changes = []
    for step_values in steps_values:
        percentage_change = [(after - before) / before * 100 for before, after in zip(original_values, step_values)]
        percentage_changes.append(percentage_change)

    # Plotting the bars for each algorithm:
    for i, percentage_change in enumerate(percentage_changes):
        label = step_labels[i] if step_labels else f'Step {i+1}'
        color = colors[i] if colors else None
        ax.bar(r1 + i * barWidth, percentage_change, width=barWidth, edgecolor='grey', label=label, color=color)

    ax.set_xticks(r1 + barWidth * (num_steps - 1) / 2)
    ax.set_xticklabels(evaluation_metrics)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel('Evaluation Metric')
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Evaluation metrics comparison (each method vs. original pipeline)')
    ax.legend()

    # Adding text on each bar (the % change for it)
    for i, percentage_change in enumerate(percentage_changes):
        for j, v in enumerate(percentage_change):
            if v < -99:
                text = f'{v:.3f}%'
            else:
                text = f'{v:.1f}%'
            if v < 0:
                ax.text(j + i * barWidth, v, text, ha='center', va='top')
            else:
                ax.text(j + i * barWidth, v, text, ha='center', va='bottom')
    
    # The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
    # plt.savefig('fullComparisonCarPrices.pdf', bbox_inches='tight')
    plt.show()



################################################################################################################

"""
    The following function will be called throughout this notebook several times (for each of the 4 datasets), 
    in order to plot a summarized evaluation metrics comparison between the following:
    (i) The performance of the model according to the original pipeline.
    (ii) The performance of the model after applying the most relevant baseline algorithm (the most relevant baseline algorithm is determined according to the sub-algorithm which our adaptive full algorithm chose to apply; if it chose to perform sub-algorithm #1 (Box-Cox + Z-Score) or sub-algorithm #3 (Outlier Capping + Z-Score), the most relevant baseline algorithm will be baseline algorithm #2 (Naive Z-Score), and if it chose to perform sub-algorithm #2 (Adaptive IQR Multiplier), the most relevant baseline algorithm will be baseline algorithm #1 (Naive IQR).
    (iii) The performance of the model after applying our adaptive algorithm (which chooses to perform a sub-algorithm out of the three).
"""

def plot_comparison(xgb_r2_original, xgb_mape_original, xgb_mae_original, xgb_rmse_original,
                    xgb_r2_baseline, xgb_mape_baseline, xgb_mae_baseline, xgb_rmse_baseline,
                    xgb_r2_best, xgb_mape_best, xgb_mae_best, xgb_rmse_best):
    # Plotting a graph comparing each of the 4 evaluation metrics for the original pipeline and the improved pipeline:
    plt.figure(figsize=(10, 10))

    # Plotting the R^2 Score
    plt.subplot(2, 2, 1)
    bars = plt.bar(['Original Pipeline', 'Most Relevant Baseline', 'The Algorithm'], 
                   [xgb_r2_original, xgb_r2_baseline, xgb_r2_best], 
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.axhline(y=min(xgb_r2_original, xgb_r2_baseline, xgb_r2_best), color='red', linestyle='--')
    plt.title('R^2 Score Comparison')
    plt.ylabel('R^2 Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=25, ha='right') 
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

    # Plotting the Mean Absolute Percentage Error (MAPE)
    plt.subplot(2, 2, 2)
    bars = plt.bar(['Original Pipeline', 'Most Relevant Baseline', 'The Algorithm'], 
                   [xgb_mape_original, xgb_mape_baseline, xgb_mape_best], 
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.axhline(y=min(xgb_mape_original, xgb_mape_baseline, xgb_mape_best), color='red', linestyle='--')
    plt.title('Mean Absolute Percentage Error (MAPE) Comparison')
    plt.ylabel('MAPE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=25, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

    # Plotting the Mean Absolute Error (MAE)
    plt.subplot(2, 2, 3)
    bars = plt.bar(['Original Pipeline', 'Most Relevant Baseline', 'The Algorithm'], 
                   [xgb_mae_original, xgb_mae_baseline, xgb_mae_best], 
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.axhline(y=min(xgb_mae_original, xgb_mae_baseline, xgb_mae_best), color='red', linestyle='--')
    plt.title('Mean Absolute Error (MAE) Comparison')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=25, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(round(yval, 0)), ha='center', va='bottom')
        

    # Plotting the Root Mean Squared Error (RMSE)
    plt.subplot(2, 2, 4)
    bars = plt.bar(['Original Pipeline', 'Most Relevant Baseline', 'The Algorithm'], 
                   [xgb_rmse_original, xgb_rmse_baseline, xgb_rmse_best],
                    color=['lightblue', 'lightcoral', 'lightgreen'])
    plt.axhline(y=min(xgb_rmse_original, xgb_rmse_baseline, xgb_rmse_best), color='red', linestyle='--')
    plt.title('Root Mean Squared Error (RMSE) Comparison')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=25, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(round(yval, 0)), ha='center', va='bottom')

    plt.tight_layout()
    # The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
    # plt.savefig('comparisonFourMetricsHouses.pdf', bbox_inches='tight')
    plt.show()


################################################################################################################

"""
    The following function shows a plot comparing the PIM score of our adaptive algorithm vs. the PIM score of the most relevant baseline:
"""

def plot_pim_comparison(baseline_score, algorithm_score, nameOfBaseline, nameOfAlgorithm):
    # Plotting a graph comparing the Percentage Improvement Metric (PIM), in %, for the algorithm and the most relevant baseline:
    plt.figure(figsize=(4.5, 6))

    # Defining labels for the plot (we run this function each time for a different algorithm, each one has a different relevant baseline algorithm):
    labels = ['Most Relevant Baseline\n(' + nameOfBaseline + ')', 'The Algorithm\n(' + nameOfAlgorithm + ')']
    values = [baseline_score, algorithm_score]

    # Plotting the bars:
    bars = plt.bar(labels, values, color=['lightblue', 'lightgreen'])

    # Adding a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Percentage Improvement Metric (PIM) Comparison')
    plt.ylabel('PIM (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjusting y-axis limits for better readability
    plt.ylim(min(values) - 10, max(values) + 10)

    # Adding text on each bar (the % change for it):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval, 2)}%', ha='center', va='bottom' if yval >= 0 else 'top')

    # Showing the plot (I also showed a legend - explanation that the dotted line represents the original pipeline)
    plt.tight_layout()
    plt.legend(['0% PIM - Original Pipeline'])
    plt.show()

################################################################################################################

"""
    The following function is similar to the 'plot_pim_comparison' function above, it also presents a comparison between the PIM score of our adaptive algorithm vs. the PIM score of the most relevant baseline,
    but this function shows 4 PIM comparison plots, each one for each of the 4 datasets which is used throughout this notebook.
"""

def plot_pim_comparison_forAllFourDatasets(baseline_score1, algorithm_score1, datasetName1, baselineName1, algorithmName1, baseline_score2, algorithm_score2, datasetName2, baselineName2, algorithmName2, baseline_score3, algorithm_score3, datasetName3, baselineName3, algorithmName3, baseline_score4, algorithm_score4, datasetName4, baselineName4, algorithmName4):
    # Defining the labels and values for each dataset, based on the parameters passed to the function:
    datasets = [
        (baseline_score1, algorithm_score1, datasetName1),
        (baseline_score2, algorithm_score2, datasetName2),
        (baseline_score3, algorithm_score3, datasetName3),
        (baseline_score4, algorithm_score4, datasetName4)
    ]

    # Creating a figure with 4 subplots side by side (one for each dataset):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for i, (baseline_score, algorithm_score, dataset_name) in enumerate(datasets):
        ax = axes[i]
        labels = ['Relevant Baseline\n(' + eval(f'baselineName{i+1}') + ')', 'The Algorithm\n(' + eval(f'algorithmName{i+1}') + ')']
        values = [baseline_score, algorithm_score]

        # Plotting the bars
        bars = ax.bar(labels, values, color=['lightblue', 'lightgreen'])
        # Adding a horizontal line at y=0 for reference (the original pipeline)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xticklabels(labels, ha='center')
        ax.set_title(f'PIM Comparison\n{dataset_name}')
        ax.set_ylabel('PIM (%)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjusting y-axis limits for better readability:
        ax.set_ylim(-58, max(values) + 10)

        # Adding text on each bar (the % change for it):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval, 2)}%', ha='center', va='bottom' if yval >= 0 else 'top')

        if i == 0:
            # Adding legend to the first subplot (there is no need to add it to all subplots, it's the same...):
            ax.legend(['0% PIM - Original Pipeline'])

    # Showing the plot:
    plt.tight_layout()
    # The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
    # plt.savefig('fourPIMPlots.pdf', bbox_inches='tight')
    plt.show()



################################################################################################################

"""
    The following code creates a plot showing both the IQR Multiplier that is used and the resulting R^2 score as a function of the correction factor (as part of sub-algorithm 2 - Adaptive IQR Multiplier):
    This will be used to show that the optimum value for the correction factor is around 0.7  (hence a correction factor of 2/3 is chosen,
    as similar optimization on other datasets showed that 2/3 3 is consistently close to the optimal factor).
"""
    
# Creating a plot to visualize the effect of the correction factor on the IQR Multiplier and the four evaluation metrics:
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plotting the IQR Multiplier on the y-axis:
ax1.plot(results_df["Factor"], results_df["IQR Multiplier"], marker='o', label="IQR Multiplier", color='b')
ax1.set_xlabel("Correction Factor")
ax1.set_ylabel("IQR Multiplier", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Creating ANOTHER y-axis for R^2 Score (one of the y-axes is on the left, and the other in on the right, and there are two lines in different colors, each for a different y-axis):
ax2 = ax1.twinx()
ax2.plot(results_df["Factor"], results_df["R^2 Score"], marker='o', label="R^2 Score", color='r')
ax2.set_ylabel("R^2 Score", color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.suptitle("Effect of the Correction Factor on the IQR Multiplier and R^2 Score (Houses Dataset)")
fig.tight_layout()
# Showing a legend for both lines:
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
# The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
# plt.savefig('correlationFactorEffect.pdf', bbox_inches='tight')
plt.show()



################################################################################################################

"""
    The following code creates a plot showing the distribution of the target variable (for example, 'charges' in the insurance dataset),
    before and after applying the Box-Cox transformation. This will be used several times, to show the effect of this transformation on the target variable.
    In particular, this will be used to demonstrate that the Box-Cox transformation is effective in reducing the number of extreme values (outliers) in the target variable.
    In addition, this will also be used to show why for the insurance dataset, performing sub-algorithm 1 (Box-Cox + Z-Score) doesn't change anything relative to the
    original pipeline (0% change), as after applying the Box-Cox transformation, no data points are outside 3 standard deviations.
"""

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plotting a KDE plot of the target variable (without Box-Cox transformation):
sns.kdeplot(y_train, label="train", ax=axes[0])
sns.kdeplot(y_test, label="test", ax=axes[0])
axes[0].set_title("Distribution of the target variable (insurance dataset)")
axes[0].set_xlabel("charges (target variable)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Plotting a KDE plot of the target variable (after peforming Box-Cox transformation):
y_train_bc, lambda_boxcox = boxcox(y_train)
y_test_bc = boxcox(y_test, lmbda=lambda_boxcox)
sns.kdeplot(y_train_bc, label="train", ax=axes[1])
sns.kdeplot(y_test_bc, label="test", ax=axes[1])
axes[1].set_xlabel("charges (Box-Cox transformed target variable)")
axes[1].set_ylabel("Density")

# Drawing red dotted lines at the mean and at +-3 standard deviations from the mean:
axes[1].axvline(y_train_bc.mean() - 3*y_train_bc.std(), color='red', linestyle='--')
axes[1].text(y_train_bc.mean() - 3*y_train_bc.std()-0.3, 0.2, "-3 std", rotation=90)
axes[1].axvline(y_train_bc.mean() + 3*y_train_bc.std(), color='red', linestyle='--')
axes[1].text(y_train_bc.mean() + 3*y_train_bc.std()+0.2, 0.2, "+3 std", rotation=90)

# Showing the plot:
axes[1].legend()
axes[1].set_title("Distribution of the target variable after Box-Cox transformation")
plt.tight_layout()
# The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
# plt.savefig('boxcoxBeforeAndAfterInsurance.pdf', bbox_inches='tight')
plt.show()

# Printing how many values are outside 3 standard deviations:
print("Number of values outside 3 standard deviations in the training set after box-cox transformation: ", len(y_train_bc[(y_train_bc < y_train_bc.mean() - 3*y_train_bc.std()) | (y_train_bc > y_train_bc.mean() + 3*y_train_bc.std())]))
print("Number of values outside 3 standard deviations in the test set after box-cox transformation: ", len(y_test_bc[(y_test_bc < y_test_bc.mean() - 3*y_test_bc.std()) | (y_test_bc > y_test_bc.mean() + 3*y_test_bc.std())]))


################################################################################################################

"""
    The following code creates a plot showing the relative PIM improvement (algorithm vs. the most relevant baseline) for all four datasets;
    The formula I use is: 100*((1+(algorithm_PIM_score_currDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_currDataset/100))-1):
    This formula gives us the percentage improvement in the PIM score of the algorithm over the most relevant baseline.
"""

# The formula I use is: 100*((1+(algorithm_PIM_score_currDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_currDataset/100))-1):
# This formula gives us the percentage improvement in the PIM score of the algorithm over the most relevant baseline.
percentage_improvement_housesDataset = 100*(((1+(algorithm_PIM_score_housesDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_housesDataset/100)))-1)
percentage_improvement_diamondsDataset = 100*((1+(algorithm_PIM_score_diamondsDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_diamondsDataset/100))-1)
percentage_improvement_insuranceDataset = 100*((1+(algorithm_PIM_score_insuranceDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_insuranceDataset/100))-1)
percentage_improvement_carsDataset = 100*((1+(algorithm_PIM_score_carsDataset/100))/(1-((-1)*most_relevant_baseline_PIM_score_carsDataset/100))-1)

# Now, we create a plot, with 4 dots - one for each dataset, showing the percentage improvement in the PIM score of the algorithm over the most relevant baseline for each dataset:
# also, there will be a line showing the average percentage improvement over the 4 datasets.
plt.figure(figsize=(10, 6))
plt.plot(["Houses (#1)", "Diamonds (#2)", "Insurance (#3)", "Car Prices (#4)"], [percentage_improvement_housesDataset, percentage_improvement_diamondsDataset, percentage_improvement_insuranceDataset, percentage_improvement_carsDataset], 'bo-', label="Improvement for each dataset (%)")

# Drawing a dotted red line showing the average percentage PIM improvement over the 4 datasets:
plt.axhline(y=(percentage_improvement_housesDataset + percentage_improvement_diamondsDataset + percentage_improvement_insuranceDataset + percentage_improvement_carsDataset)/4, color='r', linestyle='--', label="Average Improvement Across 4 datasets (%)")

# Now, for each dataset, I add a text label showing its value:
plt.text(0, percentage_improvement_housesDataset - 13, f"{percentage_improvement_housesDataset:.2f}%", color='blue', ha="center")
plt.text(1, percentage_improvement_diamondsDataset + 8, f"{percentage_improvement_diamondsDataset:.2f}%", color='blue', ha="center")
plt.text(2, percentage_improvement_insuranceDataset - 13, f"{percentage_improvement_insuranceDataset:.2f}%", color='blue', ha="center")
plt.text(3, percentage_improvement_carsDataset + 8, f"{percentage_improvement_carsDataset:.2f}%", color='blue', ha="center")

# Showing the plot:
plt.xlabel("Dataset")
plt.ylabel("PIM Improvement (%)")
plt.xticks(["Houses (#1)", "Diamonds (#2)", "Insurance (#3)", "Car Prices (#4)"])
ylabels = [-50, 0, 50, 100, 150, 200]
plt.yticks(ylabels)
# Showing text with the average percentage improvement over the 4 datasets: (with the value the dotted red line represents)
plt.text(2.05, (percentage_improvement_housesDataset + percentage_improvement_diamondsDataset + percentage_improvement_insuranceDataset + percentage_improvement_carsDataset)/4 + 5, f"Mean: {(percentage_improvement_housesDataset + percentage_improvement_diamondsDataset + percentage_improvement_insuranceDataset + percentage_improvement_carsDataset)/4:.2f}%", color='red')
plt.ylim(-50, 200)
plt.title("PIM Improvement (algorithm vs. most relevant baseline) per dataset")
plt.legend()
plt.grid()
# The following line was used in order to save the plot as a PDF file (to be added as a figure in the article):
# plt.savefig('percentageImprovementForAllDatasets.pdf', bbox_inches='tight')
plt.show()


################################################################################################################
