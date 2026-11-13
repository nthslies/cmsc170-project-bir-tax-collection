import numpy as np
import math
from filereader import load_data
from normalize import normalize_data
from evaluation import evaluate_model, make_baselines

# ------------------------------------------------------------------
# Training Naive Bayes Classifier
# ------------------------------------------------------------------
def training_classifier(x_train, y_train, k):
    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    # 2 classes of the target variable (Up or Down)
    classes = np.unique(y_train) # 1 = Up, 0 = Down
    region_col = len(x_train[0]) - 7 # index of the region column (7 = columns for [month_sin, month_cos, previous_revenue, previous_3month_mean, previous_3month_volatility, previous_month_mom_growth, historical_median_mom_growth])

    mean = {}
    variance = {}
    class_prior = {}
    p_region = {}
    for c in classes:
        x_c = x_train[y_train == c]
        region_counts = np.sum(x_c[:, :region_col], axis=0)  # count occurrences of each region for class c
        p_region[c] = (region_counts + k) / (len(x_c) + k * region_col)  # apply Laplace smoothing
        
        #Calculate Mean and Variance
        mean[c] = np.mean(x_c[:, region_col:], axis=0)
        variance[c] = np.var(x_c[:, region_col:], axis=0)

        #P(class c) = (# of instances in class c) / (total # of instances)
        class_prior[c] = len(x_c) / len(x_train)
    return classes, region_col, mean, variance, class_prior, p_region

# ------------------------------------------------------------------
# Prediction using Naive Bayes Classifier
# ------------------------------------------------------------------
def prediction(x_test, classes, region_col, mean, variance, class_prior, p_region):
    predictions = []
    for test in x_test:
        region_index = 0
        for i in range(region_col):
            if test[i] == 1:
                region_index = i
        scores = {}
        for c in classes:
            score = math.log(class_prior[c]) # score = log(P(class c))
            score = score + math.log(max(p_region[c][region_index], 1e-10)) # score = score + log(P(region | class c))

            for j in range(region_col, len(test)):
                x = test[j]
                mean_c = mean[c][j - region_col]
                variance_c = variance[c][j - region_col]
                score = score + (-0.5 * math.log(2 * math.pi * variance_c) - ((x - mean_c) ** 2) / (2 * variance_c)) # score = score + log(P(x | class c))
            scores[c] = score

        p_up_unnormalized = math.exp(scores[1])
        p_down_unnormalized = math.exp(scores[0])
        p_up = p_up_unnormalized / (p_up_unnormalized + p_down_unnormalized)

        if p_up > 0.5:
            predictions.append(1)  # Predict Up
        else:
            predictions.append(0)  # Predict Down
    return np.array(predictions, dtype = int)
            
def naive_bayes():
    x_train, y_train, x_test, y_test = load_data()
    if x_train is not None:
            # indicator that data loaded
            print("\nData Loading Completed!")
    # ------------------------------------------------------------------
    # Ask for the Smoothing Factor
    # ------------------------------------------------------------------
    while True:
        k_input = input("Enter k: ")
        try:
            k = float(k_input)
            if k > 0:
                print(f"\nk = {k}\n")
                break
            else:
                 print("Please enter a positive number for k.")
        except ValueError:
            print("Please enter a valid input")

    x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)
    classes, region_col, mean, variance, class_prior, p_region = training_classifier(x_train_scaled, y_train, k)
    predictions = prediction(x_test_scaled, classes, region_col, mean, variance, class_prior, p_region)
    # baselines use raw data (unnormalized)
    baselines = make_baselines(x_train, y_train, x_test)
    evaluate_model(y_test, predictions, baselines=baselines, model_name="Naive Bayes")
    
    return predictions

if __name__ == "__main__":
    naive_bayes()