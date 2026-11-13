# imports 
import numpy as np
from filereader import load_data
from normalize import normalize_data
from evaluation import evaluate_model, make_baselines


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
# Main
# (OwO) =============================================

def knn_run():
    # loads the tain and test files from filereader.py (kok)
    x_train, y_train, x_test, y_test = load_data()
    if x_train is None:
        return

    # scales previous revenue thru normalization
    x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)

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
    predictions = knn_predict(x_train_scaled, y_train, x_test_scaled, k, p)

    print("KNN CLASSIFICATION RESULTS")
    print(f"k = {k}")
    print(f"p = {p}\n")

    baselines = make_baselines(x_train, y_train, x_test)
    evaluate_model(y_test, predictions, baselines=baselines, model_name="KNN")
    
    return predictions

if __name__ == "__main__":
    knn_run()


    