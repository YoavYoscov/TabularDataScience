"""
    This file outlines the code which is contained in the Jupyter notebook for the baseline algorithms
    (in order to compare model's performance using the suggested adaptive algorithm vs. using each of these baselines).
"""

################################################################################################################

"""
    Baseline Algorithm #1: IQR Outlier Removal (with the generic 1.5 multiplier):
"""

def apply_iqr_outlier_removal(dtf_train, dtf_test, targetVariableName):
    # Calculating the 1st and 3rd quartiles, and the IQR accordingly:
    Q1 = dtf_train[targetVariableName].quantile(0.25)
    Q3 = dtf_train[targetVariableName].quantile(0.75)
    IQR = Q3 - Q1

    # Based on the calculated IQR, calculating the lower and upper bounds (using the generic 1.5 multiplier - the standard method):
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing the outliers (values that are more than 1.5*IQR away from the 1st and 3rd quartiles):
    dtf_train = dtf_train[(dtf_train[targetVariableName] >= lower_bound) & (dtf_train[targetVariableName] <= upper_bound)]
    return dtf_train, dtf_test


################################################################################################################

"""
    Baseline Algorithm #2: Z-Score Outlier Removal:
"""

def apply_zscore_outlier_removal(dtf_train, dtf_test, targetVariableName):
    # I calculate the z-scores of the target variable, and then remove the outliers (values that are more than 3 standard deviations away from the mean):
    dtf_train = dtf_train[(zscore(dtf_train[targetVariableName]) > -3) & (zscore(dtf_train[targetVariableName]) < 3)]
    return dtf_train, dtf_test


################################################################################################################

"""
    The Original Pipeline: NO outlier handling:
"""

def apply_no_transformation(dtf_train, dtf_test, targetVariableName):
    # This function is defined so when looping through the various outlier removal methods, I can apply the "No Transformation" method (which does nothing) - comparing to the original pipeline:
    return dtf_train, dtf_test


################################################################################################################

