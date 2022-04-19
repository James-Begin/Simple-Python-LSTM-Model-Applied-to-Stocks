import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#User enters stock:
stock = input("Enter a stock ticker: (ex. AAPL)")

#set start and end dates for training data
start = dt.datetime(2011,1,1)
end = dt.datetime(2020,1,1)
#get our data from yahoo finance
data = web.DataReader(stock, 'yahoo', start, end)

#prepare and scale data for LSTM model
#scale close price data so that it is represented by a value from 0 to 1 for use by the neural network using sklearn's MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#Lookback days
lookback = 100

#prepare the data for training by passing the # of lookback days and the
x_train = []
y_train = []

#from the lookback days to the end of the data, we add the scaled data to the training arrays.
for x in range(lookback, len(data_scaled)):
    x_train.append(data_scaled[x-lookback:x,0])
    y_train.append(data_scaled[x, 0])

#convert data to numpy arrays and reshape them
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build lstm model
model = Sequential()

#basic template of LSTM model, between each LSTM layer 20% of nodes are randomly set to 0 to prevent overfitting.
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #the prediction unit

#compile model using standard mean squared error to calculate loss
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model with batches of 32 samples for each of the 25 epochs.
model.fit(x_train, y_train, epochs=25, batch_size=32)



#predict prices from 2020 to current day
test_start = dt.datetime(2020, 1, 1,) #test on data not included in training data
test_end = dt.datetime.now()

test_data = web.DataReader(stock, 'yahoo', test_start, test_end)
real_prices = test_data['Close'].values

#combine the close prices of the real data and model data
combined_set = pd.concat((data['Close'], test_data['Close']), axis=0)

#define our model inputs and lookback days
model_input = combined_set[len(combined_set) - len(test_data) - lookback:].values #start using training data from before the timeframe so that we dont start with NaN values

model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input)

#predict based on data

test_x = []

for x in range(lookback, len(model_input)):
    test_x.append(model_input[x-lookback:x, 0]) #only append data from the lookback period

test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

predicted_prices = model.predict(test_x)
predicted_prices = scaler.inverse_transform(predicted_prices) #Since all the prices are scaled from 0 to 1 for training, convert back to prices to plot

#plotting using matplotlib
plt.plot(real_prices, color='black', label=f"real {stock} prices")
plt.plot(predicted_prices, color='red', label=f"predicted {stock} prices")
plt.title(f"Real and Predicted {stock} Share Prices")
plt.xlabel("Time")
plt.ylabel(f"{stock} Share Price")
plt.legend()
plt.show()

#Next-day Prediction

real_data = [model_input[len(model_input) + 1 - lookback:len(model_input+1), 0]]
real_data = np.array[real_data]
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#pass data through model
next_prediction = model.predict(real_data)
#convert back to prices from the scaled values
next_prediction = scaler.inverse_transform(next_prediction)
print(f"prediction: {next_prediction}")
