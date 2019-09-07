import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []


def get_data(filename):
    with open(filename, "r") as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)
        for row in csvfileReader:
            dates.append(int(row[0].split("-")[2]))
            prices.append(float(row[1]))

    return


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel="linear", C=1e3, gamma="auto")
    svr_lin.fit(dates, prices)

    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices)

    svr_sig = SVR(kernel="sigmoid", C=1e3, gamma="auto")
    svr_sig.fit(dates, prices)

    plt.scatter(dates, prices, color="black", label="Data")
    plt.plot(dates, svr_lin.predict(dates), color="red", label="Linear model")
    plt.plot(dates, svr_rbf.predict(dates), color="green", label="RBF model")
    plt.plot(dates, svr_sig.predict(dates), color="blue", label="Sigmoid model")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()

    return svr_lin.predict(np.reshape(x, (-1, 1)))[0], svr_rbf.predict(np.reshape(x, (-1, 1)))[0], svr_sig.predict(np.reshape(x, (-1, 1)))[0]


get_data("TSLA.csv")

svr_lin_pred, svr_rbf_pred, svr_sig_pred = predict_prices(dates, prices, 32)

print("Predictions:")
print(f"Linear Model: {svr_lin_pred}")
print(f"RBF model: {svr_rbf_pred}")
print(f"Sigmoid model: {svr_sig_pred}")
