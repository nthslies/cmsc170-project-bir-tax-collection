# Shared evaluation code for the modelsz

# imports
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# names of the two classes, 0 and 1
CLASS_NAMES = ["At/Below", "Above"]

# (OwO) --------------------------------------------------------------------------
# Metrics
# (OwO) --------------------------------------------------------------------------

# count the confusion matrix where:
# rows =  actual classes
# columns = predicted classes
def confusion_matrix(y_actual, y_predicted):
    tn = 0
    fp = 0
    fn = 0
    tp = 0
 
    for actual, predicted in zip(y_actual, y_predicted):
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 1 and predicted == 1:
            tp += 1
 
    return [[tn, fp], [fn, tp]]

# compute metrics from actual and predicted table
def compute_metrics(y_actual, y_predicted):
    matrix = confusion_matrix(y_actual, y_predicted)
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]
    total = tn + fp + fn + tp
 
    # zero-division
    accuracy = (tp + tn) / total if total != 0 else 0.0
 
    # for each class: (true positives, false positives, false negatives)
    # for class 0, the roles of the positive and negative class are swapped
    class_counts = [(tn, fn, fp), (tp, fp, fn)]
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for class_tp, class_fp, class_fn in class_counts:
        # -- precision (1u1)
        precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) != 0 else 0.0
        # -- recall (2o2)
        recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) != 0 else 0.0
        # -- F1-score (3_3)
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "precision": precision_scores,
        "recall": recall_scores,
        "f1": f1_scores,
        "macro_precision": sum(precision_scores)/2,
        "macro_recall": sum(recall_scores)/2,
        "macro_f1": sum(f1_scores)/2,
    }

# (OwO) --------------------------------------------------------------------------
# Baselines
# (OwO) --------------------------------------------------------------------------
# bali the baseline we mentioned in the comparison sa milestone 01


# predicts most common training class for every test row
def majority_baseline(y_train, n_test):
    label = 1 if np.mean(y_train) >= 0.5 else 0
    return np.full(n_test, label, dtype = int)

# guesses randomly, class 1 with smae share as training data
def random_baseline(y_train, n_test, seed = 170):
    # get mean (1u1)
    total = 0.0
    length = len(y_train)
    for i in range(length):
        total += y_train[i]

    share = total / length if length > 0 else 0.0

    #(2o2) random number 
    rng = random.Random(seed)

    # (3_3) preallocate result list
    result = [0] * n_test

    for i in range(n_test):
        if rng.random() < share:
            result[i] = 1
        else:
            result[i] = 0

    return np.array(result, dtype=int)

# gets the month number (1 - 12) back from month_sin and month_cos
def get_months(x, sin_col, cos_col):
    n_rows = len(x)

    # Pre allocate result array
    months = [0] * n_rows
    two_pi = 2 * math.pi

    for i in range(n_rows):
        # extract values
        sin_val = x[i][sin_col]
        cos_val = x[i][cos_col]

        angle = math.atan2(sin_val, cos_val)

        angle = angle % two_pi
        scaled = (angle / two_pi) * 12

        rounded = int(round(scaled))

        month = rounded % 12

        if month == 0:
            month = 12

        months[i] = month

    return np.array(months)

# predicts the most common training class of the same month using raw features
# predicts that whichever label happened the most during a given month in training will happen again in the same month in testing
def month_baseline(x_train, y_train, x_test, sin_col = 18, cos_col = 19):

    # convert the sine and cosine features into discrete month values
    train_months = get_months(x_train, sin_col, cos_col)
    test_months = get_months(x_test, sin_col, cos_col)

    # fallback prediction frmo overall majority class
    overall = 1 if np.mean(y_train) >= 0.5 else 0

    month_label = {}

    # determines most frequent label for each month from training
    for month in range(1, 13):
        labels = y_train[train_months == month]
        if len(labels) > 0:
            month_label[month] = 1 if np.mean(labels) >= 0.5 else 0
        else:
            month_label[month] = overall

    return np.array([month_label[month] for month in test_months], dtype = int)

# Gather all baselines in one direction
def make_baselines(x_train, y_train, x_test, sin_col = 18, cos_col = 19):
    return {
        "Always majority": majority_baseline(y_train, len(x_test)),
        "Random Class Ratio": random_baseline(y_train, len(x_test)),
        "Month Only": month_baseline(x_train, y_train, x_test, sin_col, cos_col)
    }

# (OwO) --------------------------------------------------------------------------
# Printing
# (OwO) --------------------------------------------------------------------------

# to be called by each model (knn, naive, perceptron)
def evaluate_model(y_actual, y_predicted, baselines = None, model_name = "Model", class_names = CLASS_NAMES):
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)
    results = compute_metrics(y_actual, y_predicted)
    matrix = np.array(results["confusion_matrix"])

    # displays confusion matrix
    print("\n-----------CONFUSION MATRIX-----------")
    print(f"{'':16}Predicted")
    print(f"{'':16}{class_names[0]:>7}{class_names[1]:>7}")
    print(f"Actual {class_names[0]:<9}{matrix[0][0]:7d}{matrix[0][1]:7d}")
    print(f"Actual {class_names[1]:<9}{matrix[1][0]:7d}{matrix[1][1]:7d}")

     # displays evaluation metrics
    print("\n----------EVALUATION METRICS----------")
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall:    {results['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {results['macro_f1']:.4f}")

    # displays metrics for each class with the counts behind them
    print("\n-----------PER-CLASS METRICS-----------")
    for i in range(2):
        predicted_count = int(np.sum(matrix[:, i]))
        actual_count = int(np.sum(matrix[i, :]))
        print(f"\nClass: {class_names[i]}")
        print(f"Precision: {results['precision'][i]:.4f} ({matrix[i][i]}/{predicted_count})")
        print(f"Recall:    {results['recall'][i]:.4f} ({matrix[i][i]}/{actual_count})")
        print(f"F1-Score:  {results['f1'][i]:.4f}")

    # displays the baselines next to the model
    if baselines:
        print("\n-----------BASELINE COMPARISON-----------")
        print(f"{model_name:<24}{results['accuracy']:>10.4f}{results['macro_f1']:>10.4f}")
        results["baselines"] = {}
        for name, predictions in baselines.items():
            baseline_results = compute_metrics(y_actual, predictions)
            results["baselines"][name] = baseline_results
            print(f"{name:<24}{baseline_results['accuracy']:>10.4f}{baseline_results['macro_f1']:>10.4f}")

    return results

# (OwO) --------------------------------------------------------------------------
# Model Plotting #https://www.geeksforgeeks.org/python/matplotlib-axes-axes-bar-in-python/
# (OwO) -------------------------------------------------------------------------- 
def plot_model(rows, folder):
    models = []
    accuracies = []
    macro_f1s = []

    for row in rows:
        if row[1] == "model" and "one-hot" in row[2]:
            if row[0] == "Naive Bayes":
                continue
            models.append(f"{row[0]} ({row[2]})")
            accuracies.append(float(row[4]))
            macro_f1s.append(float(row[7]))
    if not models:
        print("No models found to plot.")
        return
    data = list(zip(accuracies, macro_f1s))
    metrics = ["Accuracy", "Macro F1"]
    colors = ["purple", "lavender"]
    
    dim = len(data[0])
    w = 0.6
    dimw = w / dim

    fig, ax = plt.subplots()
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        b = ax.bar(x + i * dimw, y, 
                dimw,
                color = colors[i],
                label = metrics[i],
                bottom = 0.001)

    ax.set_xticks(x + w / 2 - dimw / 2)
    ax.set_xticklabels(models)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.legend(metrics)

    save_figure(fig, f"{folder}/model_performance.png")
    plt.close(fig)
# (OwO) --------------------------------------------------------------------------
# Naive Bayes Plotting
# (OwO) -------------------------------------------------------------------------- 
def plot_naive_bayes(rows, folder):
    models = []
    accuracies = []
    macro_f1s = []

    for row in rows:
        if row[1] == "model":
            if "Naive Bayes" in row[0]:
                if "one-hot" in row[2]:
                    models.append(f"{row[0]} ({row[2]})")
                    accuracies.append(float(row[4]))
                    macro_f1s.append(float(row[7]))
    if not models:
        print("No models found to plot.")
        return
    data = list(zip(accuracies, macro_f1s))
    metrics = ["Accuracy", "Macro F1"]
    colors = ["purple", "lavender"]
    
    dim = len(data[0])
    w = 0.6
    dimw = w / dim

    fig, ax = plt.subplots()
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        b = ax.bar(x + i * dimw, y, 
                dimw,
                color = colors[i],
                label = metrics[i],
                bottom = 0.001)

    ax.set_xticks(x + w / 2 - dimw / 2)
    ax.set_xticklabels(models)
    ax.set_title("Naive Bayes Performance Comparison")
    ax.set_ylabel("Score")
    ax.legend(metrics)

    save_figure(fig, f"{folder}/model_naivebayes.png")
    plt.close(fig)

# added from kmeans
# saves a figure, if the file cannot be written (for example it is open in a viewer) it is saved under a new name
def save_figure(figure, path):
  try:
    figure.savefig(path, dpi=150)
  except OSError as error:
    fallback = path[:-4] + "_new.png"
    print(f"Could not save {path} ({error}). Close the file if it is open in another program.")
    try:
      figure.savefig(fallback, dpi=150)
      print(f"Saved it as {fallback} instead.")
    except OSError as second_error:
      print(f"Could not save {fallback} either ({second_error}).")

# (OwO) --------------------------------------------------------------------------
# KNN Tuning Plotting
# (OwO) --------------------------------------------------------------------------

# KEEP copy of colors heeeere (O-O)
LINE_COLORS = ["#6A5AB0", "#9B8EC7", "#B4D3D9"]
LINE_MARKERS = ["o", "s", "^"]
EDGE_COLOR = "#5B5780"

# line chart of the validation macro f1 (top row) and macro precision (bottom row)
# x axis is k, one line for each p, one column for each month weight
# details is the list that tune_knn fills when it gets details = []
# test_results is the compute_metrics result of the chosen setting on the 2024 test data, it is drawn as a dashed line
def plot_knn_tuning(details, folder, tag, feature_label, test_results = None):
    from matplotlib.lines import Line2D

    if not details:
        print("No tuning results to plot.")
        return

    # gets every value of each factor, finds the best row (the first best one wins ties, same as the tuning)
    p_values = []
    k_values = []
    weights = []
    lookup = {}
    best = details[0]
    for row in details:
        if row["p"] not in p_values:
            p_values.append(row["p"])
        if row["k"] not in k_values:
            k_values.append(row["k"])
        if row["month_weight"] not in weights:
            weights.append(row["month_weight"])
        lookup[(row["p"], row["k"], row["month_weight"])] = row
        if row["macro_f1"] > best["macro_f1"]:
            best = row
    p_values.sort()
    k_values.sort()
    weights.sort()

    metrics = [("macro_f1", "Macro-F1"), ("macro_precision", "Macro Precision")]
    x_positions = list(range(len(k_values)))
    best_index = p_values.index(best["p"])

    fig, axes = plt.subplots(2, len(weights), figsize = (max(8.0, 4.4 * len(weights) + 1.2), 7.2), sharex = True, sharey = "row", squeeze = False)

    for row_index in range(2):
        metric, metric_name = metrics[row_index]

        # same y range for the whole row
        all_values = [row[metric] for row in details]
        if test_results is not None:
            all_values.append(test_results[metric])
        low = min(all_values) - 0.03
        high = max(all_values) + 0.03
        # a little more room above the test line for its label
        if test_results is not None:
            high = max(all_values) + 0.055

        for column in range(len(weights)):
            weight = weights[column]
            ax = axes[row_index][column]

            # one line for each p
            for i in range(len(p_values)):
                p = p_values[i]
                values = []
                for k in k_values:
                    values.append(lookup[(p, k, weight)][metric])
                ax.plot(x_positions, values, color = LINE_COLORS[i % 3], marker = LINE_MARKERS[i % 3],
                        markersize = 6, markeredgecolor = EDGE_COLOR, markeredgewidth = 0.8,
                        linewidth = 2, label = f"p = {p:g}")

            # bigger point on the best macro f1 setting (shown in both rows so the precision there can be read)
            if weight == best["month_weight"]:
                best_x = k_values.index(best["k"])
                best_y = lookup[(best["p"], best["k"], weight)][metric]
                ax.plot([best_x], [best_y], color = LINE_COLORS[best_index % 3], marker = LINE_MARKERS[best_index % 3],
                        markersize = 14, markeredgecolor = EDGE_COLOR, markeredgewidth = 1.2, linestyle = "none", zorder = 5)

                # the score of the same setting on the 2024 test data, trained on all the years before 2024
                if test_results is not None:
                    test_y = test_results[metric]
                    ax.axhline(test_y, color = EDGE_COLOR, linestyle = "--", linewidth = 1.2, zorder = 1)
                    ax.plot([best_x, best_x], [best_y, test_y], color = EDGE_COLOR, linestyle = ":", linewidth = 1.2, zorder = 2)
                    ax.plot([best_x], [test_y], color = EDGE_COLOR, marker = "D", markersize = 9,
                            markeredgecolor = "white", markeredgewidth = 1.0, linestyle = "none", zorder = 6)
                    ax.text(len(k_values) - 0.6, test_y + 0.004, f"2024 test: {test_y:.4f}", ha = "right", va = "bottom", fontsize = 9, color = EDGE_COLOR)

            ax.set_ylim(low, high)
            ax.set_xlim(-0.5, len(k_values) - 0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(k) for k in k_values])

            if row_index == 0:
                if feature_label == "base":
                    ax.set_title("month_sin / month_cos (no month weight)")
                else:
                    ax.set_title(f"Month weight = {weight}")
            if column == 0:
                ax.set_ylabel(f"{metric_name} (validation mean)")
            if row_index == 1:
                ax.set_xlabel("k (number of neighbors)")

    # one legend, the p lines and what the bigger point is
    handles, labels = axes[0][0].get_legend_handles_labels()
    best_label = f"best Macro-F1: k = {best['k']}, p = {best['p']:g}"
    if feature_label != "base":
        best_label += f", month weight = {best['month_weight']}"
    handles.append(Line2D([0], [0], color = LINE_COLORS[best_index % 3], marker = LINE_MARKERS[best_index % 3], markersize = 12,
                          markeredgecolor = EDGE_COLOR, markeredgewidth = 1.2, linestyle = "none"))
    labels.append(best_label)
    if test_results is not None:
        handles.append(Line2D([0], [0], color = EDGE_COLOR, linestyle = "--", marker = "D", markersize = 8,
                              markeredgecolor = "white", markeredgewidth = 1.0))
        labels.append("2024 test score at the best setting")
    legend_columns = len(handles)
    legend_space = 0.05
    if len(weights) == 1:
        legend_columns = 3
        legend_space = 0.09
    fig.legend(handles, labels, loc = "lower center", ncol = legend_columns, frameon = False)

    if feature_label == "base":
        fig.suptitle("KNN Tuning: Macro-F1 and Macro Precision by k and p")
    else:
        fig.suptitle("KNN Tuning: Macro-F1 and Macro Precision by k, p and month weight")
    fig.tight_layout(rect = (0, legend_space, 1, 0.97))

    os.makedirs(folder, exist_ok = True)
    save_figure(fig, f"{folder}/knn_tuning_{tag}.png")
    plt.close(fig)

# (OwO) --------------------------------------------------------------------------
# Main
# (OwO) --------------------------------------------------------------------------
# runs all classification models on each feature set and writes their results in a csv

# -- ETO YUN CHANGES --
# "base" is the original features (bali yung based sa preprocessed data with sin and cos columns from csv)
#  "month one-hot" replaces month_sin and month_cos with 12 month columns (bali ilabel as 1 or 0 whetehr active column na yon)
# remove one of them to run only the other, so if both, irarun both in sequence
FEATURE_SETS = ["base", "month one-hot"]
# if TUNE_KNN is True, k, p and the month weight are picked on the validation folds
# and KNN_K, KNN_P, KNN_MONTH_WEIGHT are not used
TUNE_KNN = True
KNN_K = 7
KNN_P = 2.0
KNN_MONTH_WEIGHT = 2
NAIVE_BAYES_SMOOTHING = 1.0
LIDSTONE_SMOOTHING = 0.001
RESULTS_FILE = "classification_results.csv"

# makes one csv row from the results of a model or a baseline
# tn, fp, fn, tp are counted with class 1 as the positive class
def result_row(name, kind, features, settings, results):
    matrix = results["confusion_matrix"]
    row = [name, kind, features, settings]
    row.append(f"{results['accuracy']:.4f}")
    row.append(f"{results['macro_precision']:.4f}")
    row.append(f"{results['macro_recall']:.4f}")
    row.append(f"{results['macro_f1']:.4f}")
    # precision, recall, f1 of each class
    for i in range(2):
        row.append(f"{results['precision'][i]:.4f}")
        row.append(f"{results['recall'][i]:.4f}")
        row.append(f"{results['f1'][i]:.4f}")
    row.extend([matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]])
 
    return row

def main():
    # imports are inside main so the models can import this file without a circular import
    import csv
    import io
    import contextlib
    from filereader import load_data
    from normalize import normalize_data, add_month_onehot, REGION_COLS, MONTH_COLS
    from knn import knn_predict, tune_knn, show_tuning, weight_month
    from naivebayes import training_classifier, prediction
    from perceptron import train_perceptron, predict_perceptron
 
    # loads data once for all the models
    x_train, y_train, x_test, y_test, years_train = load_data(return_years = True)
    if x_train is None:
        return
 
    # the baselines always use the raw original features
    baselines = make_baselines(x_train, y_train, x_test)
 
    rows = []
    for feature_set in FEATURE_SETS:
        print(f"\n######## FEATURES: {feature_set.upper()} ########")
        # (1u1) chooses the features of this run
        if feature_set == "month one-hot":
            x_train_used = add_month_onehot(x_train)
            x_test_used = add_month_onehot(x_test)
            month_cols = (REGION_COLS, REGION_COLS + MONTH_COLS)
            group_sizes = [REGION_COLS, MONTH_COLS]
        else:
            x_train_used = x_train
            x_test_used = x_test
            month_cols = None
            group_sizes = None
        tag = feature_set.replace(" ", "_").replace("-", "")

        # scales the features
        x_train_scaled, x_test_scaled = normalize_data(x_train_used, x_test_used)
 
        predictions = {}
        settings = {}
 
        # KNN, k, p and the month weight are picked on the validation folds
        knn_k = KNN_K
        knn_p = KNN_P
        knn_weight = KNN_MONTH_WEIGHT
        knn_note = ""
        knn_details = [] # filled by tune_knn, used for the tuning chart
        if TUNE_KNN:
            print("\nTuning KNN on the validation folds...")
            knn_k, knn_p, knn_weight, tuning_results, validate_years = tune_knn(x_train_used, y_train, years_train, month_cols = month_cols, details = knn_details)
            show_tuning(tuning_results, validate_years, knn_k, knn_p, knn_weight, f"knn_tuning_{tag}.csv")
            knn_note = " (chosen by validation)"

        x_train_knn = x_train_scaled
        x_test_knn = x_test_scaled
        knn_setting = f"k={knn_k}; p={knn_p}"
        if month_cols is not None:
            x_train_knn = weight_month(x_train_scaled, month_cols, knn_weight)
            x_test_knn = weight_month(x_test_scaled, month_cols, knn_weight)
            knn_setting += f"; month weight={knn_weight}"
        print("\nRunning KNN...")
        predictions["KNN"] = knn_predict(x_train_knn, y_train, x_test_knn, knn_k, knn_p)
        # the tuning chart also shows the 2024 test score of the chosen setting
        if TUNE_KNN:
            plot_knn_tuning(knn_details, "results/graphs", tag, feature_set, compute_metrics(y_test, predictions["KNN"]))
        settings["KNN"] = knn_setting + knn_note
 
        print("Running Naive Bayes...")
        nb_model = training_classifier(x_train_scaled, y_train, NAIVE_BAYES_SMOOTHING, group_sizes)
        predictions["Naive Bayes"] = prediction(x_test_scaled, *nb_model)
        settings["Naive Bayes"] = f"smoothing k={NAIVE_BAYES_SMOOTHING}"

        # after further testing lidstone smoothing with a very small k gave better results than the default k=1
        nbl_model = training_classifier(x_train_scaled, y_train, LIDSTONE_SMOOTHING, group_sizes)
        predictions["Naive Bayes (Lidstone)"] = prediction(x_test_scaled, *nbl_model)
        settings["Naive Bayes (Lidstone)"] = f"smoothing k={LIDSTONE_SMOOTHING}"
 
        print("Training Perceptron...")
        # the training prints every epoch, so the printing is hidden here
        with contextlib.redirect_stdout(io.StringIO()):
            weights = train_perceptron(x_train_scaled, y_train)
        predictions["Perceptron"] = predict_perceptron(x_test_scaled, weights)
        settings["Perceptron"] = "see perceptron.py"
 
        # (2o2) shows and collects the results of each model
        for name in predictions:
            print(f"\n================ {name.upper()} [{feature_set}] ================")
            results = evaluate_model(y_test, predictions[name], baselines = baselines, model_name = name)
            rows.append(result_row(name, "model", feature_set, settings[name], results))
 
    # (3_3) collects the results of the baselines
    for name, baseline_predictions in baselines.items():
        rows.append(result_row(name, "baseline", "raw features", "", compute_metrics(y_test, baseline_predictions)))
 
    header = ["method", "type", "features", "settings", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for class_name in CLASS_NAMES:
        header.extend([f"precision_{class_name}", f"recall_{class_name}", f"f1_{class_name}"])
    header.extend(["tn", "fp", "fn", "tp"])
 
    with open(RESULTS_FILE, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    plot_model(rows, "results/graphs")
    plot_naive_bayes(rows, "results/graphs")

    # shows a summary of everything that was saved
    print("\n================ SUMMARY ================")
    print(f"{'Method':<22}{'Features':<16}{'Accuracy':>10}{'Macro F1':>10}")
    for row in rows:
        print(f"{row[0]:<22}{row[2]:<16}{row[4]:>10}{row[7]:>10}")
    print(f"\nResults saved to {RESULTS_FILE}")
 
if __name__ == "__main__":
    main()