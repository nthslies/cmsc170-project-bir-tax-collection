# imports 
import csv
import numpy as np
from filereader import load_data
from normalize import normalize_data
from evaluation import evaluate_model, make_baselines, compute_metrics

# values tried when tuning, k must be odd and smaller than the smallest training fold
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
P_VALUES = [1, 2, 3]
# how much the month columns count in the distance, only used with the month one-hot features
MONTH_WEIGHTS = [1, 2, 3]
TUNING_FILE = "knn_tuning.csv"

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

# for each test sample, gets the labels of all training points from the nearest to the farthest
# the distances are computed only once, so many values of k can use them (u-u)
def nearest_labels(x_train, y_train, x_test, p):
    all_labels = []

    for test_sample in x_test:
        # calculate distance to each training point
        distances = []
        for train_sample, train_label in zip(x_train, y_train):
            d = minkowski(test_sample, train_sample, p)
            # stored as (distance, label)
            distances.append((d, train_label))

        # sort distances from the nearest
        distances.sort()

        # keeps only the labels
        sorted_labels = []
        for d, label in distances:
            sorted_labels.append(label)
        all_labels.append(sorted_labels)

    return all_labels

# majority vote of the k nearest labels, 1 is up and 0 is down
def vote(sorted_labels, k):
    # tally the lebels of k nearst neighbors (o_o)
    up = 0
    down = 0
    for i in range(k):
        neighbor_label = sorted_labels[i]
        if neighbor_label == 1:
            up += 1
        else:
            down += 1

    # choose by majority vote
    if up > down:
        return 1
    return 0

# prediction thru knn (u-u)
def knn_predict(x_train, y_train, x_test, k, p):
    all_labels = nearest_labels(x_train, y_train, x_test, p)

    predictions = []
    for sorted_labels in all_labels:
        predictions.append(vote(sorted_labels, k))

    return np.array(predictions, dtype = int)

# (OwO) =============================================
# Tuning
# (OwO) =============================================

# multiplies the month columns by a weight so the month counts more in the distance
# month_cols is (first month column, last month column + 1)
def weight_month(x, month_cols, weight):
    start, end = month_cols
    # np.array makes a copy so the original is not changed
    weighted = np.array(x, dtype = float)
    for i in range(len(weighted)):
        for j in range(start, end):
            weighted[i][j] = weighted[i][j] * weight

    return weighted

# gets the rows of a list at the given positions
def pick_rows(data, indices):
    rows = []
    for i in indices:
        rows.append(data[i])

    return rows

# makes the forward chaining folds, each fold trains on all the years before the year it validates
# first_validate_year skips the earliest folds because they have too little training data
# every fold is (validate year, positions of the training rows, positions of the validation rows)
def year_folds(years_train, first_validate_year = 2022):
    all_years = sorted(set(years_train))
    folds = []

    for validate_year in all_years:
        if validate_year < first_validate_year:
            continue
        train_indices = []
        validate_indices = []
        for i in range(len(years_train)):
            if years_train[i] < validate_year:
                train_indices.append(i)
            elif years_train[i] == validate_year:
                validate_indices.append(i)
        folds.append((validate_year, train_indices, validate_indices))

    return folds

# tries every p, k and month weight on the validation folds and picks the best by mean macro f1
# x_train must be the raw features, each fold is normalized with its own training rows only
# month_cols is (first month column, last month column + 1) when the features have the month one-hot columns, else None
def tune_knn(x_train, y_train, years_train, k_values = K_VALUES, p_values = P_VALUES, month_cols = None, month_weights = MONTH_WEIGHTS):
    folds = year_folds(years_train)

    # without month columns there is nothing to weight
    if month_cols is None:
        month_weights = [1]

    # only keeps the k that are smaller than the smallest training fold
    smallest_train = len(folds[0][1])
    for validate_year, train_indices, validate_indices in folds:
        if len(train_indices) < smallest_train:
            smallest_train = len(train_indices)
    usable_k = []
    for k in k_values:
        if k <= smallest_train:
            usable_k.append(k)

    # every result is [p, k, month weight, score of each fold..., mean score]
    results = []
    for p in p_values:
        for weight in month_weights:
            scores = {}
            for k in usable_k:
                scores[k] = []

            for validate_year, train_indices, validate_indices in folds:
                # splits and normalizes using the fold's training rows only
                x_fold_train = np.array(pick_rows(x_train, train_indices), dtype = float)
                y_fold_train = pick_rows(y_train, train_indices)
                x_fold_validate = np.array(pick_rows(x_train, validate_indices), dtype = float)
                y_fold_validate = pick_rows(y_train, validate_indices)
                x_fold_train, x_fold_validate = normalize_data(x_fold_train, x_fold_validate)
                if month_cols is not None:
                    x_fold_train = weight_month(x_fold_train, month_cols, weight)
                    x_fold_validate = weight_month(x_fold_validate, month_cols, weight)

                # distances are computed once for all k
                all_labels = nearest_labels(x_fold_train, y_fold_train, x_fold_validate, p)
                for k in usable_k:
                    predictions = []
                    for sorted_labels in all_labels:
                        predictions.append(vote(sorted_labels, k))
                    scores[k].append(compute_metrics(y_fold_validate, predictions)["macro_f1"])

            for k in usable_k:
                mean_score = sum(scores[k]) / len(scores[k])
                results.append([p, k, weight] + scores[k] + [mean_score])

    # picks the highest mean, the first one wins ties (smaller p, then smaller weight, then smaller k)
    best = results[0]
    for row in results:
        if row[-1] > best[-1]:
            best = row

    validate_years = []
    for validate_year, train_indices, validate_indices in folds:
        validate_years.append(validate_year)

    return best[1], best[0], best[2], results, validate_years

# prints the tuning results and saves them on a csv file
def show_tuning(results, validate_years, best_k, best_p, best_weight, file_name = TUNING_FILE):
    header = ["p", "k", "month_weight"]
    for year in validate_years:
        header.append(f"validate_{year}")
    header.append("mean_macro_f1")

    print("\n-----------KNN TUNING (macro F1 on validation folds)-----------")
    line = f"{'p':<6}{'k':<6}{'month w':<9}"
    for year in validate_years:
        line += f"{year:<10}"
    line += "mean"
    print(line)
    for row in results:
        line = f"{row[0]:<6}{row[1]:<6}{row[2]:<9}"
        for score in row[3:-1]:
            line += f"{score:<10.4f}"
        line += f"{row[-1]:.4f}"
        print(line)
    print(f"\nBest: k = {best_k}, p = {best_p}, month weight = {best_weight}")

    with open(file_name, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in results:
            writer.writerow([row[0], row[1], row[2]] + [f"{score:.4f}" for score in row[3:]])
    print(f"Tuning results saved to {file_name}")

# (OwO) =============================================
# Main
# (OwO) =============================================

def knn_run():
    # loads the tain and test files from filereader.py (kok)
    x_train, y_train, x_test, y_test, years_train = load_data(return_years = True)
    if x_train is None:
        return

    # scales previous revenue thru normalization
    x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)

    # gets k from user, odd numbers only, 0 means pick k and p from the validation folds
    k = int(input("\nEnter an odd value of k (0 to tune k and p): "))
    while k < 0 or (k != 0 and k % 2 == 0) or k > len(x_train):
        print(f"Please input 0 or an odd number from 1 to {len(x_train)}!")
        k = int(input("Enter an odd value of k (0 to tune k and p): "))

    if k == 0:
        print("\nTuning on the validation folds...")
        k, p, weight, results, validate_years = tune_knn(x_train, y_train, years_train)
        show_tuning(results, validate_years, k, p, weight)
    else:
        # gets p from user
        p = float(input("Enter the value of p for Minkowski distance: "))
        while p < 1:
            print("p must be at least 1!")
            p = float(input("Enter the value of p for Minkowski distance: "))

    # (uwu) getting the predictions
    print("\nClassifying test instances...\n")
    predictions = knn_predict(x_train_scaled, y_train, x_test_scaled, k, p)

    print("KNN CLASSIFICATION RESULTS")
    print(f"k = {k}")
    print(f"p = {p}\n")

    baselines = make_baselines(x_train, y_train, x_test)
    evaluate_model(y_test, predictions, baselines=baselines, model_name="KNN")
    
    return predictions

if __name__ == "__main__":
    knn_run()
