# simple_stock_prediction
A scikit-learn application, including simple stock predictions from three types of regression models.

The method used by this application is to take the last 29 days of a stock price as input and then ouput the predicted price for the 30th day.
The latest data is used as test data, while the rest is used for training. 
This may decrease the accuracy of the models on the test data, 
since stock prices from let's say 7 years ago aren't as similar to the latest prices of a stock as stock prices closer to that time.
So a random distribution of test data and training data would probably be a good idea to implement.

Other machine learning models such as LSTM's may be better fitted for jobs like these.
