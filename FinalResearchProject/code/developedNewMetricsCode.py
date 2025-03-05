"""
    This file outlines the code which is contained in the Jupyter notebook for the two new developed metrics I introduced as part of the project;
    these metrics are used throughout the project, as well as more thoroughly explained in the PDF article for this project.
"""

################################################################################################################

"""
    Percentage Improvement Metric (PIM): as part of this project, I suggest a combined metric/score I developed,
    which averages the percentage improvement of the four evaluation metrics (R^2, MAE, MAPE, RMSE) relative to the original baseline.
    Percentage improvement enables combining metrics of different value ranges/trends ($R^2$ has opposite trend than the other three).
    The formula I devised for PIM is introduced both as part of the PDF article for this project and the Jupyter notebook examplifying my solution, as well as implemented below:
"""

def calculatePIM(xgb_r2_originalPipeline, xgb_mape_originalPipeline, xgb_mae_originalPipeline, xgb_rmse_originalPipeline,
                                     xgb_r2_bestMethod, xgb_mape_bestMethod, xgb_mae_bestMethod, xgb_rmse_bestMethod,
                                     xgb_r2_mostRelevantBaseline, xgb_mape_mostRelevantBaseline, xgb_mae_mostRelevantBaseline, xgb_rmse_mostRelevantBaseline):
    # Calculating the Percentage Improvement Metric (PIM), in %, for the algorithm vs. the original pipeline:
    improvement_algorithm = [
        -25 * (1 - (xgb_r2_bestMethod / xgb_r2_originalPipeline)),
        25 * (1 - (xgb_mape_bestMethod / xgb_mape_originalPipeline)),
        25 * (1 - (xgb_mae_bestMethod / xgb_mae_originalPipeline)),
        25 * (1 - (xgb_rmse_bestMethod / xgb_rmse_originalPipeline))
    ]
    algorithm_PIM_score = sum(improvement_algorithm)

    # Calculating the Percentage Improvement Metric (PIM), in %, for the most relevant baseline vs. the original pipeline:
    improvement_most_relevant_baseline = [
        -25 * (1 - (xgb_r2_mostRelevantBaseline / xgb_r2_originalPipeline)),
        25 * (1 - (xgb_mape_mostRelevantBaseline / xgb_mape_originalPipeline)),
        25 * (1 - (xgb_mae_mostRelevantBaseline / xgb_mae_originalPipeline)),
        25 * (1 - (xgb_rmse_mostRelevantBaseline / xgb_rmse_originalPipeline))
    ]
    most_relevant_baseline_PIM_score = sum(improvement_most_relevant_baseline)
    
    return algorithm_PIM_score, most_relevant_baseline_PIM_score

################################################################################################################

"""
    Normalized Variance (NV):
    I have defined a metric called “Normalized Variance” (NV) which is used as part of the decision rules of the adaptive data-driven algorithm:
    This metric is calculated based on the variance and mean of the dataset (the default value for zero mean is 0).
    If NV < 0.5: low variance - extreme values are not frequent, don’t carry important information and can be removed as outliers - sub-algorithm 2.
    If NV > 0.5: high variance - extreme values are quite frequent and cannot be removed; capping is more suitable - sub-algorithm 3
"""

def getNormalizedVariance(df, target_col):
    data = df[target_col]

    variance = np.var(data)
    mean = np.mean(data)
    meanSquared = mean**2
    if mean != 0:
        normalizedVariance = variance / meanSquared
    else:
        normalizedVariance = 0

    return normalizedVariance

################################################################################################################

"""
    This function gets a specific outlier-handling method to perform, then applies it, trains the model, and returns the evaluation metrics accordingly.
    This will be used to evaluate the effect of each outlier-handling method on the model's performance:
"""

def train_and_evaluate(dtf_train, dtf_test, method_name, targetVariableName):
    # Splitting the data into X and y:
    X_train = dtf_train.drop([targetVariableName], axis=1)
    X_test = dtf_test.drop([targetVariableName], axis=1)
    y_train = dtf_train[targetVariableName]
    y_test = dtf_test[targetVariableName]
    
    # Training the model (XGBoost):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # The four evaluation metrics used:
    metrics = {
        "Method": method_name,
        "R^2 Score": r2_score(y_test, predictions),
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }
    
    return metrics

# Also defining the 6 different variations of the outlier removal methods we are going to perform for each dataset, this will be the basis for the experimental evaluation
# (thus enabling the comparison of the adaptive algorithm (with the sub-algorithm it has selcted), and veify that it indeed outperforms all other algorithms -
# both the original pipeline, the two baseline algorithms, and the other two sub-algorithms):
methods = {
    "Original - no outlier handling": apply_no_transformation,
    "Naive IQR": apply_iqr_outlier_removal,
    "Naive Z-Score": apply_zscore_outlier_removal,
    "Box-Cox + Z-Score": apply_boxcox_transformation,
    "Adaptive IQR Multiplier": apply_iqr_withAdaptiveMultiplier,
    "Capping with Z-Score": apply_capping_zscore
}



################################################################################################################
