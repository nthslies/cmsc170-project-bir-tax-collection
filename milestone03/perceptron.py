# imports
import csv
import numpy as np
from filereader import load_data
from normalize import normalize_data
from evaluation import evaluate_model, make_baselines, CLASS_NAMES


# function to train perceptron
def train_perceptron(x_train, y_train):
    # converts labels from {0,1} to {1,0} for perceptron training
    y_train = np.where(y_train == 0, -1, 1)
    r = 0.85 # learning rate set to 0.85
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

# function to save the evaluation results on the same csv file
def save_results(results):
    confusion_matrix = results["confusion_matrix"]
    with open("perceptron_training.csv", "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["CONFUSION MATRIX"])
        writer.writerow(["", f"Predicted {CLASS_NAMES[0]}", f"Predicted {CLASS_NAMES[1]}"])
        writer.writerow([f"Actual {CLASS_NAMES[0]}", confusion_matrix[0][0], confusion_matrix[0][1]])
        writer.writerow([f"Actual {CLASS_NAMES[1]}", confusion_matrix[1][0], confusion_matrix[1][1]])
        writer.writerow([])
        writer.writerow(["EVALUATION METRICS"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{results['accuracy']:.4f}"])
        writer.writerow(["Macro Precision", f"{results['macro_precision']:.4f}"])
        writer.writerow(["Macro Recall", f"{results['macro_recall']:.4f}"])
        writer.writerow(["Macro F1-Score", f"{results['macro_f1']:.4f}"])
        writer.writerow([])
        writer.writerow(["PER-CLASS METRICS"])
        writer.writerow(["Class", "Precision", "Recall", "F1-Score"])

        for i in range(2):
            writer.writerow([
                CLASS_NAMES[i],
                f"{results['precision'][i]:.4f}",
                f"{results['recall'][i]:.4f}",
                f"{results['f1'][i]:.4f}"
            ])

    print("Results saved to perceptron_training.csv")


def main():
    # loads data
    x_train, y_train, x_test, y_test = load_data()
    # checks if data loading failed
    if x_train is None:
        return
    # normalizes data
    x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)
    # train perceptron
    print("Training Perceptron")
    w = train_perceptron(x_train_scaled, y_train)
    # predict testing data
    y_predicted = predict_perceptron(x_test_scaled, w)
    # baselines using unnormalized data
    baselines = make_baselines(x_train, y_train, x_test)
    # evaluate predictions
    results = evaluate_model(y_test, y_predicted, baselines=baselines, model_name="Perceptron")
    save_results(results)

if __name__ == "__main__":
    main()