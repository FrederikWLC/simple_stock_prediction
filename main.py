import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime


# Function for getting the data from the csv_file
def get_data(filename):
    dates = []
    data = [[], []]

    with open(filename, "r") as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)
        X = []
        for i, row in enumerate(csvfileReader):
            i += 1

            # Only for every 30th price; add that price as the y-value in the training data
            if i % 30 == 0 and i != 0:
                dates.append(datetime.strptime(row[0], "%Y-%m-%d"))
                data[0].append(X)
                data[1].append(float(row[1]))
                X = []
            else:
                # Else; add that price to the X-array in the training data
                X.append(float(row[1]))

        # Give the last 15 sets of data to the test_data and the rest to train_data
        test_data = [data[0][-15:], data[1][-15:]]
        train_data = [data[0][:-15], data[1][:-15]]
        test_dates = dates[-15:]
        train_dates = dates[:-15]
    return train_data, train_dates, test_data, test_dates


# Generates and trains the models
def gen_models(train_data):
    svr_lin = SVR(kernel="linear", C=1e3, gamma="auto")
    svr_lin.fit(train_data[0], train_data[1])

    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
    svr_rbf.fit(train_data[0], train_data[1])

    svr_sig = SVR(kernel="sigmoid", C=1e3, gamma="auto")
    svr_sig.fit(train_data[0], train_data[1])

    return svr_lin, svr_rbf, svr_sig


# Plots the models' fitness on data
def plot(svr_lin, svr_rbf, svr_sig, data, dates, title):
    plt.scatter(dates, data[1], color="black", label="Data")
    plt.plot(dates, svr_lin.predict(data[0]), color="red", label="Linear model")
    plt.plot(dates, svr_rbf.predict(data[0]), color="green", label="RBF model")
    plt.plot(dates, svr_sig.predict(data[0]), color="blue", label="Sigmoid model")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Support Vector Regression ({title})")
    plt.legend()
    plt.show()


# Predict next price when given 29 earlier prices
def predict_prices(model, X):
    return model.predict([X])[0]


# Getting the data
train_data, train_dates, test_data, test_dates = get_data("TSLA.csv")

# Generating and training the models
svr_lin, svr_rbf, svr_sig = gen_models(train_data)

# Plot the models' fitness on the training data
plot(svr_lin, svr_rbf, svr_sig, train_data, train_dates, title="Predictions on training data")

# Plot the models' fitness on the test data
plot(svr_lin, svr_rbf, svr_sig, test_data, test_dates, title="Predictions on test data")

# Predicting and printing the last price in test data
svr_lin_pred = predict_prices(svr_lin, test_data[0][-1])
svr_rbf_pred = predict_prices(svr_rbf, test_data[0][-1])
svr_sig_pred = predict_prices(svr_sig, test_data[0][-1])
print(f"Predictions of stock price at ({test_dates[-1]}):\n")
print(f"Linear Model: {svr_lin_pred}")
print(f"RBF model: {svr_rbf_pred}")
print(f"Sigmoid model: {svr_sig_pred}")
print(f"The actual stock price: {test_data[1][-1]}")
