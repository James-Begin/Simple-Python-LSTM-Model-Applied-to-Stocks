# Simple-Python-LSTM-Model-Applied-to-Stocks
Using sklearn and tensorflow, this project allows the user to choose a stock, and a lookback window. The data is fetched from Yahoo Finance and passed through a basic LSTM model which then estimates the next day's stock price.
The data is first organized and prepared for modelling. Then, the model is trained over 25 epochs (can take 10+ mins without dedicated GPU). 
The model is a basic LSTM template with 3 LSTM layers, each with 50 units each and dropping out 20% of the input units between each layer. Finally, the dense layer returns the output in one dimension.

Once trained, the model is used to predict the next day's stock price from the start of 2020 to current day. These predictions are plotted and compared to the actual stock price of that day. Then, the next day's prediction is returned.

![LSTModelpythonFB](https://user-images.githubusercontent.com/103123677/162008654-b8593316-0dfb-45d6-aa16-e6da18e1334e.png)

#**Dependencies:**
>NumPy: pip install numpy\
>matplotlib: pip install matplotlib\
>pandas: pip install pandas\
>pandas-datareader: pip install pandas-datareader\
>tensorflow: pip install tensorflow\
>scikit learn: pip install scikit-learn\
  
