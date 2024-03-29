# simple_stock_prediction
A scikit-learn application, including simple stock predictions from three types of regression models.

The method used by this application is to take the last 29 days of a stock price as input and then ouput the predicted price for the 30th day.
The latest data is used as test data, while the rest is used for training. 
This may decrease the accuracy of the models given the test data as input, 
since stock prices from, let's say 7 years ago, aren't as similar to the latest prices of a stock as stock prices only months away from the latest prices.
So a random distribution of test data and training data would probably be a good idea to implement.

Other machine learning models such as LSTM's may be better fit for jobs like these, as they are much more efficient at memorizing past information.
