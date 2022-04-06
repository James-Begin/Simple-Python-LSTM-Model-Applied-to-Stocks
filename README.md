# Simple-Python-LSTM-Model-Applied-to-Stocks
Using sklearn and tensorflow, this project allows the user to choose a stock, and a lookback window. The data is fetched from Yahoo Finance and passed through a basic LSTM model which then estimates the next day's stock price.
The data is first organized and prepared for modelling. Then, the model is trained over 25 epochs (can take 10+ mins without dedicated GPU).
