import pandas as pd
import yfinance as yf
import numpy as np

class StockPair:
    def __init__(self, ticker1, ticker2, start_date='2020-01-01', end_date='2023-01-01'):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def load_data(self):
        # Download historical data for the two stocks
        self.data = yf.download([self.ticker1, self.ticker2], start=self.start_date, end=self.end_date)
        print(self.data.head())  # Display the first few rows of data

    def get_close_prices(self):
        if self.data is not None:
            self.close_price = self.data['Close']
            print(self.close_price.head())  # Display the first few rows of close_price
            return self.close_price
        else:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
    def cointegration_test(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        log_price1 = np.log(self.data[f'{self.ticker1}']) 
        log_price2 = np.log(self.data[f'{self.ticker2}'])


