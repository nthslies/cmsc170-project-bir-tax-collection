# imports 
import numpy as np
from filereader import load_data

# (OwO) =============================================
# Normalization
# (OwO) =============================================
# function for normalizing features between 0 and 1
def normalize_data(x_train, x_test):
    # gets minimum and maximum for each feature
    x_min = np.min(x_train, axis = 0)
    x_max = np.max(x_train, axis = 0)
    x_range = x_max - x_min
    # handles division by zero
    x_range[x_range == 0] = 1.0
    # normalize train and test
    x_train = (x_train - x_min) / x_range
    x_test = (x_test - x_min) / x_range

    return x_train, x_test

# (OwO) =============================================
# Distance and Prediction
# (OwO) =============================================

# summation of the absolute value of xi - vi raised to p
# all multiplied by 1/p
def minkowski(x, v, p):
  total = 0.0
  for i in range(len(x)):
    d = abs(x[i] - v[i]) #absolute value
    total += d ** p

  return total ** (1/p)

# prediction thru knn (u-u)
def knn_predict(x_train, y_train, x_test, k, p):
    predictions = []

    for test_sample in x_test:
        # calculate distance to each training point
        distances = []
        for train_sample, train_label in zip(x_train, y_train):
            d = minkowski(test_sample, train_sample, p)
            # stored as (distance, label)
            distances.append((d, train_label))

        # sort distances from the nearest
        distances.sort()

        # tally the lebels of k nearst neighbors (o_o)
        up = 0
        down = 0
        for i in range(k):
            neighbor_label = distances[i][1]
            if neighbor_label == 1:
                up += 1
            else:
                down += 1

        # choose by majority vote
        if up > down:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions, dtype = int)

# (OwO) =============================================
# Evaluation Metrics
# (OwO) =============================================

# counts the confusion matrix, 1 (increase) is the positive class
def confusion_counts(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    return tp, tn, fp, fn

# compute accuracy, precision, recall, and f1 score (pop)
def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn

    # zero-division handling for all metrics
    accuracy = (tp + tn) / total if total != 0 else 0.0

    # for each class: (true positives, false positives, false negatives)
    # for down, roles of the positive and negative class are swapped
    class_counts = [(tn, fn, fp), (tp, fp, fn)]
    precision = []
    recall = []
    f1 = []

    for class_tp, class_fp, class_fn in class_counts:
        curr_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) != 0 else 0.0
        curr_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) != 0 else 0.0
        if (curr_precision + curr_recall) != 0:
            curr_f1 = 2 * curr_precision * curr_recall / (curr_precision + curr_recall)
        else:
            curr_f1 = 0.0

        precision.append(curr_precision)
        recall.append(curr_recall)
        f1.append(curr_f1)
        
    return accuracy, precision, recall, f1


# (OwO) =============================================
# Main
# (OwO) =============================================

def knn_run():
    # loads the tain and test files from filereader.py (kok)
    x_train, y_train, x_test, y_test = load_data()
    if x_train is None:
        return

    # scales previous revenue thru normalization
    x_train, x_test = normalize_data(x_train, x_test)

    # gets k from user, odd numbers only
    k = int(input("\nEnter an odd value of k: "))
    while k % 2 == 0 or k < 1 or k > len(x_train):
        print(f"Please input an odd number from 1 to {len(x_train)}!")
        k = int(input("Enter an odd value of k: "))

    # gets p from user
    p = float(input("Enter the value of p for Minkowski distance: "))
    while p < 1:
        print("p must be at least 1!")
        p = float(input("Enter the value of p for Minkowski distance: "))

    # (uwu) getting the predictions
    print("\nClassifying test instances...\n")
    predictions = knn_predict(x_train, y_train, x_test, k, p)

    # getting metrics and matrix (o-o)
    tp, tn, fp, fn = confusion_counts(y_test, predictions)
    accuracy, precision, recall, f1 = compute_metrics(y_test, predictions)

    # macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)


    print("KNN CLASSIFICATION RESULTS")
    print(f"k = {k}")
    print(f"p = {p}\n")

    # confusin matrix
    print("\n-----------CONFUSION MATRIX-----------")
    print("                 Predicted")
    print("                 Down   Up")
    print(f"Actual Down      {tn:4d} {fp:4d}")
    print(f"Actual Up        {fn:4d} {tp:4d}")

    # eval metrics
    print("\n----------EVALUATION METRICS----------")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-Score:  {macro_f1:.4f}")

    # per class metrics
    print("\n-----------PER-CLASS METRICS-----------")
    class_names = ["Down", "Up"]
    for i in range(2):
        print(f"\nClass: {class_names[i]}")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall:    {recall[i]:.4f}")
        print(f"F1-Score:  {f1[i]:.4f}")

    

    return predictions

if __name__ == "__main__":
    knn_run()


    