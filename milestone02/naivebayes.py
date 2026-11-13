import numpy as np
import math
from filereader import load_data

# ------------------------------------------------------------------
# Training Naive Bayes Classifier
# ------------------------------------------------------------------
def training_classifier(x_train, y_train, k):
    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    # 2 classes of the target variable (Up or Down)
    classes = np.unique(y_train) # 1 = Up, 0 = Down
    region_col = len(x_train[0]) - 3 # index of the region column (3 = columns for [month_sin, month_cos, previous_revenue])

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

# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------
def evaluate_model(y_test, predictions):
    # initialize confusion matrix where rows = actual classes, columns = predicted classes
        confusion_matrix = np.zeros((2, 2), dtype = int)
        # counts actual and predicted
        for i in range(len(y_test)):
            actual = y_test[i]
            predicted = predictions[i]
            confusion_matrix[actual][predicted] += 1
        
        # total correct predictions
        correct = confusion_matrix[0][0] + confusion_matrix[1][1]
        # total predictions
        total = np.sum(confusion_matrix)
        # accuracy
        accuracy = correct / total if total != 0 else 0
        precision_scores = []
        recall_scores = []
        f1_scores = []
    
        # calculate metrics for each class
        for class_id in range(2):
            tp = confusion_matrix[class_id][class_id]
            fp = np.sum(confusion_matrix[:, class_id]) - tp
            fn = np.sum(confusion_matrix[class_id, :]) - tp
            # precision
            precision = (tp / (tp + fp)) if (tp + fp != 0) else 0
            # recall
            recall = (tp / (tp + fn)) if (tp + fn != 0) else 0
            # F1-score
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
    
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
    
        # macro averages
        macro_precision = np.mean(precision_scores)
        macro_recall = np.mean(recall_scores)
        macro_f1 = np.mean(f1_scores)
    
        # displays confusion matrix
        print("\n-----------CONFUSION MATRIX-----------")
        print("                 Predicted")
        print("                 Down   Up")
        print(f"Actual Down      {confusion_matrix[0][0]:4d} {confusion_matrix[0][1]:4d}")
        print(f"Actual Up        {confusion_matrix[1][0]:4d} {confusion_matrix[1][1]:4d}")
    
        # displays evaluation metrics
        print("\n----------EVALUATION METRICS----------")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1-Score:  {macro_f1:.4f}")
    
        # displays metrics for each class
        print("\n-----------PER-CLASS METRICS-----------")
        class_names = ["Down", "Up"]
        for i in range(2):
            print(f"\nClass: {class_names[i]}")
            print(f"Precision: {precision_scores[i]:.4f}")
            print(f"Recall:    {recall_scores[i]:.4f}")
            print(f"F1-Score:  {f1_scores[i]:.4f}")
            
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

    classes, region_col, mean, variance, class_prior, p_region = training_classifier(x_train, y_train, k)
    predictions = prediction(x_test, classes, region_col, mean, variance, class_prior, p_region)
    evaluate_model(y_test, predictions)
    
    return predictions
if __name__ == "__main__":
    naive_bayes()