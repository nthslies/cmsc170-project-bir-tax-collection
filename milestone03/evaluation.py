# Shared evaluation code for the modelsz

# imports
import math
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
        if row[1] == "model":
            print("DEBUG ROW:", row)
            models.append(row[0])
            accuracies.append(float(row[3]))
            macro_f1s.append(float(row[6]))
    if not models:
        print("No models found to plot.")
        return
    data = list(zip(accuracies, macro_f1s))
    metrics = ["Accuracy", "Macro F1"]
    colors = ["blue", "orange"]
    
    dim = len(data[0])
    w = 0.6
    dimw = w / dim

    fig, ax = plt.subplots()
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        b = ax.bar(x + i * dimw, y, 
                dimw, 
                bottom = 0.001)

    ax.set_xticks(x + w / 2 - dimw / 2)
    ax.set_xticklabels(models)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.legend(metrics)

    save_figure(fig, f"{folder}/model_performance.png")
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
# Main
# (OwO) --------------------------------------------------------------------------

KNN_K = 1
KNN_P = 2.0
NAIVE_BAYES_SMOOTHING = 1.0
RESULTS_FILE = "classification_results.csv"

# makes one csv row from the results of a model or a baseline
# tn, fp, fn, tp are counted with class 1 as the positive class
def result_row(name, kind, settings, results):
    matrix = results["confusion_matrix"]
    row = [name, kind, settings]
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
    from normalize import normalize_data
    from knn import knn_predict
    from naivebayes import training_classifier, prediction
    from perceptron import train_perceptron, predict_perceptron
 
    # loads data once for all the models
    x_train, y_train, x_test, y_test = load_data()
    if x_train is None:
        return
 
    # scales the features, the raw features are kept for the month baseline
    x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)
    baselines = make_baselines(x_train, y_train, x_test)
 
    # (1u1) runs every model on the same data
    predictions = {}
    settings = {}
 
    print("\nRunning KNN...")
    predictions["KNN"] = knn_predict(x_train_scaled, y_train, x_test_scaled, KNN_K, KNN_P)
    settings["KNN"] = f"k={KNN_K}; p={KNN_P}"
 
    print("Running Naive Bayes...")
    nb_model = training_classifier(x_train_scaled, y_train, NAIVE_BAYES_SMOOTHING)
    predictions["Naive Bayes"] = prediction(x_test_scaled, *nb_model)
    settings["Naive Bayes"] = f"smoothing k={NAIVE_BAYES_SMOOTHING}"
 
    print("Training Perceptron...")
    # the training prints every epoch, so the printing is hidden here
    with contextlib.redirect_stdout(io.StringIO()):
        weights = train_perceptron(x_train_scaled, y_train)
    predictions["Perceptron"] = predict_perceptron(x_test_scaled, weights)
    settings["Perceptron"] = "see perceptron.py"
 
    # (2o2) shows and collects the results of each model
    rows = []
    for name in predictions:
        print(f"\n================ {name.upper()} ================")
        results = evaluate_model(y_test, predictions[name], baselines = baselines, model_name = name)
        rows.append(result_row(name, "model", settings[name], results))
 
    # (3_3) collects the results of the baselines
    for name, baseline_predictions in baselines.items():
        rows.append(result_row(name, "baseline", "", compute_metrics(y_test, baseline_predictions)))
 
    header = ["method", "type", "settings", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for class_name in CLASS_NAMES:
        header.extend([f"precision_{class_name}", f"recall_{class_name}", f"f1_{class_name}"])
    header.extend(["tn", "fp", "fn", "tp"])
 
    with open(RESULTS_FILE, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    plot_model(rows, "results/graphs")

    # shows a summary of everything that was saved
    print("\n================ SUMMARY ================")
    print(f"{'Method':<24}{'Accuracy':>10}{'Macro F1':>10}")
    for row in rows:
        print(f"{row[0]:<24}{row[3]:>10}{row[6]:>10}")
    print(f"\nResults saved to {RESULTS_FILE}")
 
if __name__ == "__main__":
    main()