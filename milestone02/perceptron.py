# imports
import csv
import numpy as np
from filereader import load_data

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

# function to train perceptron
def train_perceptron(x_train, y_train):
    # converts labels from {0,1} to {1,0} for perceptron training
    y_train = np.where(y_train == 0, -1, 1)
    r = 0.75 # learning rate set to 0.1
    b = 0.5 # bias
    # prepares training data
    data = []
    for i in range(len(x_train)):
        x_features  = list(x_train[i])
        d = float(y_train[i])
        data.append((x_features, d))
    # determines number of features
    num_features = len(data[0][0])
    # weight vectors
    w = np.zeros(1 + num_features) 
    w[0] = b

    print(f"Learning Rate = {r}")
    print(f"Bias, w0 = {w[0]}")
    # records status of convergence and epoch
    epoch = 0
    converged = False
    with open("perceptron_training.csv", "w", newline = "") as file:
        writer = csv.writer(file)
        # learning rate initial bias
        writer.writerow([f"Learning Rate = {r}"])
        writer.writerow([f"Bias, w0 = {b}"])
        writer.writerow([])
    
        # repeats until no updates made
        while not converged and epoch < 100:
            epoch += 1
            print(f"\nEpoch: {epoch}")
            writer.writerow([f"Epoch: {epoch}"])
            # column headers for epoch
            header = []
            for i in range(num_features + 1):
                header.append(f"x{i}")
            for i in range(num_features + 1):
                header.append(f"w{i}")
            header.extend(["v", "y", "d"])
            writer.writerow(header)
            epoch_update = 0
            # for activating perceptron
            for x_features, d in data:
                x_full = np.array([1.0] + x_features)
                v = np.dot(w, x_full)
                y = 1.0 if v > 0 else -1.0
                row = []
                # adds input feature values, current weight values, weighted sum, predicted and desired output
                for value in x_full:
                    row.append(f"{value:.2f}")
                for weight in w:
                    row.append(f"{weight:.3f}")
                row.append(f"{v:.3f}")
                row.append(f"{y:.3f}")
                row.append(f"{d:.3f}")
                # saves to csv
                writer.writerow(row)
                # updates weight vector
                if y != d:
                    w = w + r * (d - y) * x_full
                    epoch_update += 1
                
            print(f"Weight updates: {epoch_update}")
            # saves weight updates for current epoch
            writer.writerow([f"Weight updates: {epoch_update}"])
            writer.writerow([])
            # checks convergence
            if epoch_update == 0:
                converged = True
                break

        # saves final weights and convergence status
        writer.writerow(["Final Adjusted Weights"])
        writer.writerow([f"{weight:.4f}" for weight in w])
        if converged:
            writer.writerow(["Training converged"])
        else:
            writer.writerow(["Training did not converge within 100 epochs"])

        # final weights
        print("\nFinal Adjusted Weight: " + ", ".join([f"{weight:.4f}" for weight in w]))
        # tells if data converged
        if converged:
            print("Training converged")
        else:
            print("Training did not converge within 100 epochs")

        return w

# function to predict perceptron
def predict_perceptron(x_test, w):
    predictions = []
    for x in x_test:
        x_features = list(x) # converts features to list
        x_full = np.array([1.0] + x_features)
        v = np.dot(w, x_full)
        prediction = 1 if v > 0 else 0
        predictions.append(prediction)

    return np.array(predictions, dtype = int)

# function to evaluate based on metrics chosen
def evaluate_model(y_actual, y_predicted):
    # initialize confusion matrix where rows = actual classes, columns = predicted classes
    confusion_matrix = np.zeros((2, 2), dtype = int)
    # counts actual and predicted
    for i in range(len(y_actual)):
        actual = y_actual[i]
        predicted = y_predicted[i]
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

    # saves evaluation metrics on same csv file
    with open("perceptron_training.csv", "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["CONFUSION MATRIX"])
        writer.writerow(["", "Predicted Down", "Predicted Up"])
        writer.writerow(["Actual Down", confusion_matrix[0][0], confusion_matrix[0][1]])
        writer.writerow(["Actual Up", confusion_matrix[1][0], confusion_matrix[1][1]])
        writer.writerow([])
        writer.writerow(["EVALUATION METRICS"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{accuracy:.4f}"])
        writer.writerow(["Macro Precision", f"{macro_precision:.4f}"])
        writer.writerow(["Macro Recall", f"{macro_recall:.4f}"])
        writer.writerow(["Macro F1-Score", f"{macro_f1:.4f}"])
        writer.writerow([])
        writer.writerow(["PER-CLASS METRICS"])
        writer.writerow(["Class", "Precision", "Recall", "F1-Score"])

        for i in range(2):
            writer.writerow([
                class_names[i],
                f"{precision_scores[i]:.4f}",
                f"{recall_scores[i]:.4f}",
                f"{f1_scores[i]:.4f}"
            ])

    print("Results saved to perceptron_training.csv")

def main():
    # loads data
    x_train, y_train, x_test, y_test = load_data()
    # checks if data loading failed
    if x_train is None:
        return
    # normalizes data
    x_train, x_test = normalize_data(x_train, x_test)
    # train perceptron
    print("Training Perceptron")
    w = train_perceptron(x_train, y_train)
    # predict testing data
    y_predicted = predict_perceptron(x_test, w)
    # evaluate predictions
    evaluate_model(y_test, y_predicted)

if __name__ == "__main__":
    main()