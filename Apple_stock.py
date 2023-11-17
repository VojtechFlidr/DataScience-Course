import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf
#Get the stock quote
df = yf.download('AAPL', start='2012-01-01', end='2023-11-15')

#Show the data
df
#Get the number of rows and columns in the data set
df.shape

#Visualize the closing price history
#We create a plot with name 'Close Price History'
plt.figure(figsize=(16,8))
plt.title('Close Price History')
#We give the plot the data (the closing price of our stock)
plt.plot(df['Close'])
#We label the axis
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#We show the plot
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Assuming 'df' is your original DataFrame

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
training_data_len = math.ceil(len(dataset) * 0.8)
train_data = scaled_data[0:training_data_len, :]

# Create x_train and y_train data sets
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # Will contain 60 values (0-59)
    y_train.append(train_data[i, 0])  # Will contain the 61st value (60)

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:]
x_test = []

# Forecast future values for the next 'future_steps' days
future_steps = 30  # Adjust as needed
for i in range(60, 60 + future_steps):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values for the x_test data set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame for the future dates and predicted prices
future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]
future_predictions = pd.DataFrame(index=future_dates, columns=['Predictions'])
future_predictions['Predictions'] = predictions

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = np.nan  # Add NaN for the existing data
valid = pd.concat([valid, future_predictions])
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()
