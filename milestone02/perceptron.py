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
    # converts labels from {0,1} to {1,0} to ensure correct weight updates
    y_train = np.where(y_train == 0, -1, 1)
    r = 0.1 # learning rate set to 0.1
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

if __name__ == "__main__":
    main()