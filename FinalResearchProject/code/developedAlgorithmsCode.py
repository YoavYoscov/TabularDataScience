"""
    This file outlines the code which is contained in the Jupyter notebook for the developed algorithms.
"""

################################################################################################################

"""
    The final adaptive, data-driven algorithm (The Yoscovich Data-Driven Predicted Best of Three Algorithm):
"""

def select_outlier_method(df, target_col, toPrint):
    data = df[target_col]

    # Calculating the skewness of the data:
    S = skew(data)

    # Calculating the normalized variance (NV, the new metric I've defined) of the data:
    variance = np.var(data)
    mean = np.mean(data)
    meanSquared = mean**2
    if mean != 0:
        normalizedVariance = variance / meanSquared
    else:
        normalizedVariance = 0

    if toPrint:
        print("Skewness: ", S)
        print("Normalized Variance: ", normalizedVariance)

    # Decision rule #1:
    # If the skewness is greater than 3, this means the data is highly skewed.
    if abs(S) > 3:
        return "Box-Cox + Z-Score"

    # Decision rule #2:
    # If the normalized variance is less than 0.5, this means the data is not too spread out.
    if normalizedVariance < 0.5:
        return "Adaptive IQR Multiplier"
    
    # Decision rule #3:
    # If the normalized variance is >= 0.5, this means the data is spread out (with potential meaningful information).
    else:
        return "Capping with Z-Score"
    
    
################################################################################################################

"""
    Sub-algorithm 1: Box-Cox + Z-Score algorithms:
"""

def apply_boxcox_transformation(dtf_train, dtf_test, targetVariableName):
    # Shifting to non-negative values: (since Box-Cox can only handle positive values)
    if (dtf_train[targetVariableName] <= 0).any():
        shift = abs(dtf_train[targetVariableName].min()) + 1
        dtf_train[targetVariableName] += shift
        dtf_test[targetVariableName] += shift
    else:
        shift = 0
    
    # Applying Box-Cox transformation, and getting the lambda value:
    dtf_train["targetVariableBoxcox"], lambda_boxcox = boxcox(dtf_train[targetVariableName])
    dtf_test["targetVariableBoxcox"] = boxcox(dtf_test[targetVariableName], lmbda=lambda_boxcox)
    
    # Calculating the z-scores of the target variable:
    z_scores = zscore(dtf_train["targetVariableBoxcox"])
    # Using the Z-Score method to remove outliers (values that are 3 standard deviations away from the mean):
    dtf_train = dtf_train[(z_scores > -3) & (z_scores < 3)]
    
    # Now, inversing the Box-Cox transformation (using the lambda value saved earlier):
    dtf_train[targetVariableName] = ((dtf_train["targetVariableBoxcox"] * lambda_boxcox + 1) ** (1 / lambda_boxcox))
    dtf_test[targetVariableName] = ((dtf_test["targetVariableBoxcox"] * lambda_boxcox + 1) ** (1 / lambda_boxcox))

    # Shifting back:
    dtf_train[targetVariableName] -= shift
    dtf_test[targetVariableName] -= shift
    
    # Note that before returning, I dropped the targetVariableBoxcox column (this was only used for the Box-Cox transformation calculations):
    return dtf_train.drop(columns=["targetVariableBoxcox"]), dtf_test.drop(columns=["targetVariableBoxcox"])


################################################################################################################

"""
    Sub-algorithm 2: Adaptive IQR Multiplier algorithm:
"""

def compute_iqr_multiplier(data):
    # Calculating basic statistics - skewness and kurtosis of the data:
    skew_value = stats.skew(data)
    kurt_value = stats.kurtosis(data, fisher=True)


    # Calculating the Mean-to-Median Ratio:
    mean_value = np.mean(data)
    median_value = np.median(data)
    if median_value != 0:
        mean_median_ratio = mean_value / median_value
    else:
        mean_median_ratio = 1


    # Calculating IQR=Q3-Q1 (1st and 3rd quartiles):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    # Also calculating the 1st and 99th percentiles:
    p1, p99 = np.percentile(data, [1, 99])
    # Using the 1st and 99th percentiles calculated above, we can now calculate the Tail Spread:
    if iqr != 0:
        tail_spread = (p99 - p1) / iqr
    else:
        tail_spread = 0
    
    # Now, I also calculate the Outlier Rate (values that are 1.5*IQR away from the 1st and 3rd quartiles):
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    outlier_rate = outliers / len(data)
    
    
    # Using the developed IQR Multiplier Formula introduced in the paper:
    iqr_multiplier = 1.5 + (2/3)*(0 
                      + abs(skew_value) 
                      + max(0, kurt_value - 3) 
                      + max(0, tail_spread - 3) 
                      + max(0, outlier_rate - 0.05)  
                      + max(0, mean_median_ratio - 1.5))

    iqr_multiplier = max(0.5, min(iqr_multiplier, 5))  
    return round(iqr_multiplier, 2)

# The following function applies the IQR method with the adaptive multiplier (it calls the function above which calculates the multiplier, and then applies the IQR method accordingly):
def apply_iqr_withAdaptiveMultiplier(dtf_train, dtf_test, targetVariableName):
    iqr_multiplier = compute_iqr_multiplier(dtf_train[targetVariableName])
    
    Q1 = dtf_train[targetVariableName].quantile(0.25)
    Q3 = dtf_train[targetVariableName].quantile(0.75)
    IQR = Q3 - Q1

    # It is important to note the use of 'iqr_multiplier', which is the calculated multiplier (according to the formula explained in the article), rather than using the generic 1.5 multiplier:
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    dtf_train = dtf_train[(dtf_train[targetVariableName] >= lower_bound) & (dtf_train[targetVariableName] <= upper_bound)]
    return dtf_train, dtf_test


################################################################################################################

"""
    Sub-algorithm 3: Outlier Capping + Z-Score algorithm:
"""

def apply_capping_zscore(dtf_train, dtf_test, targetVariableName):
    # First, calculating the z-scores of the target variable:
    z_scores = zscore(dtf_train[targetVariableName])
    # I create a mask which basically "masks" - tells which values are more than 3 standard deviations away from the mean (these are the outliers):
    mask = (z_scores < -3) | (z_scores > 3)

    # Calculating the 1st and 99th percentiles of the target variable:
    lower_bound = dtf_train[targetVariableName].quantile(0.01)
    upper_bound = dtf_train[targetVariableName].quantile(0.99)

    # This is important - the capping is done for the values that are more than 3 standard deviations away from the mean:
    # So, the outlier detection is done using the Z-Score method, but the capping is done using the 1st and 99th percentiles:
    dtf_train.loc[mask, targetVariableName] = dtf_train.loc[mask, targetVariableName].clip(lower=lower_bound, upper=upper_bound)
    return dtf_train, dtf_test

################################################################################################################

"""
    The following code is relevant for sub-algorithm 2 (Adaptive IQR Multiplier algorithm); it is used both in the Jupyter notebook and in the pdf article,
    to show the effect of the correction factor which is used (inside the adaptive IQR multiplier formula I have devised) on the model's performance.
    This shows that (2/3) is a consistent and effective correction factor to use in the formula (tested across various datasets):
"""

def compute_iqr_multiplierWithSpecificCorrectionFactor(data, multiplier):
    # Calculating basic statistics - skewness and kurtosis of the data:
    skew_value = stats.skew(data)
    kurt_value = stats.kurtosis(data, fisher=True)

    # Calculating the Mean-to-Median Ratio:
    mean_value = np.mean(data)
    median_value = np.median(data)     
    if median_value != 0:
        mean_median_ratio = mean_value / median_value
    else:
        mean_median_ratio = 1


    # Calculating IQR=Q3-Q1 (1st and 3rd quartiles):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    # Also calculating the 1st and 99th percentiles:
    p1, p99 = np.percentile(data, [1, 99])
    # Using the 1st and 99th percentiles calculated above, we can now calculate the Tail Spread:
    if iqr != 0:
        tail_spread = (p99 - p1) / iqr
    else:
        tail_spread = 0


    # Now, I also calculate the Outlier Rate (values that are 1.5*IQR away from the 1st and 3rd quartiles):
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    outlier_rate = outliers / len(data)

    
    # Using the developed IQR Multiplier Formula introduced in the paper:
    iqr_multiplier = 1.5 + multiplier * (
                      + abs(skew_value) 
                      + max(0, kurt_value - 3) 
                      + max(0, tail_spread - 3) 
                      + max(0, outlier_rate - 0.05)  
                      + max(0, mean_median_ratio - 1.5))
    return round(iqr_multiplier, 2)


def apply_iqr_withTailoredMultiplier(dtf_train, dtf_test, multiplier):
    # First, getting the calculated tailor-made IQR multiplier using the function above:
    iqr_multiplier = compute_iqr_multiplierWithSpecificCorrectionFactor(dtf_train[targetVariableName], multiplier)
    
    # Calculating the 1st and 3rd quartiles, and the IQR accordingly:
    Q1 = dtf_train[targetVariableName].quantile(0.25)
    Q3 = dtf_train[targetVariableName].quantile(0.75)
    IQR = Q3 - Q1

    # Based on the calculated IQR, calculating the lower and upper bounds (using the calculated multiplier):
    # It is important to note the use of 'iqr_multiplier', which is the calculated multiplier (according to the formula explained in the article), rather than using the generic 1.5 multiplier:
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Now, removing the outliers (values that are more than iqr_multiplier*IQR away from the 1st and 3rd quartiles):
    dtf_train = dtf_train[(dtf_train[targetVariableName] >= lower_bound) & (dtf_train[targetVariableName] <= upper_bound)]
    return dtf_train, dtf_test, iqr_multiplier


def train_and_evaluateWithFactor(dtf_train, dtf_test, factor):
    # Splitting the data into X and y:
    X_train = dtf_train.drop([targetVariableName], axis=1)
    X_test = dtf_test.drop([targetVariableName], axis=1)
    y_train = dtf_train[targetVariableName]
    y_test = dtf_test[targetVariableName]
    
    # Training the model (XGBoost):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # The four evaluation metrics used, and also the factor used for the IQR method:
    metrics = {
        "Factor": factor,
        "R^2 Score": r2_score(y_test, predictions),
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }
    
    return metrics

# Defining a list with the multipliers I want to test:
multipliers = [0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, (2/3), 0.7, 0.8, 0.9, 1]
results = []
# I will iterate over all these multipliers, and for each one, I will apply the IQR method with the tailored multiplier (and save the resulting evaluation metric scores):
for multiplier in multipliers:
    modified_train, modified_test, total_iqr_multiplier = apply_iqr_withTailoredMultiplier(dtf_train.copy(), dtf_test.copy(), multiplier)
    # Important - I call the function 'train_and_evaluateWithFactor' which is the same as 'train_and_evaluate', but also uses the specified multiplier for IQR method!
    metrics = train_and_evaluateWithFactor(modified_train, modified_test, f"{multiplier:.2f}")
    metrics["IQR Multiplier"] = total_iqr_multiplier
    # Saving the results for the current multiplier:
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df = results_df[['Factor', 'IQR Multiplier', 'R^2 Score', 'MAPE', 'MAE', 'RMSE']]
print(results_df)

################################################################################################################