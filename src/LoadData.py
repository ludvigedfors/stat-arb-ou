import pandas as pd
import yfinance as yf
import numpy as np

# StockPair:
# def __init__(self, ticker1, ticker2, start_date='2020-01-01', end_date='2023-01-01'):
#     self.ticker1 = ticker1
#     self.ticker2 = ticker2
#     self.start_date = start_date
#     self.end_date = end_date
#     self.close_price = self.load_data()
#     self.cointegration_result = self.cointegration_test()

def load_data(A = 'GOOG', B = 'GOOGL', start_date='2020-01-01', end_date='2023-01-01'):
    # Download historical data for the two stocks
    data = yf.download([A, B], start=start_date, end=end_date)
    # print(data.head())  # Display the first few rows of data
    close_price = data['Close']
    # print(close_price.head())  # Display the first few rows of close_price
    return close_price  
    

    
def cointegration_test(data):
    if data is None:
        raise ValueError("Data not loaded. Please call load_data() first.")

    ticker1 = data.columns[0]
    ticker2 = data.columns[1]

    log_pair = np.log(data).dropna()

    # OLS: B_t = alpha + beta * A_t + epsilon_t
    A = log_pair[ticker1].to_numpy()
    B = log_pair[ticker2].to_numpy()
    A_design = np.column_stack([np.ones(len(A)), A])

    alpha, beta = np.linalg.lstsq(A_design, B, rcond=None)[0]
    epsilon = B - (alpha + beta * A)

    result = pd.DataFrame({"log_A": A,"log_B": B,"fitted_B": alpha + beta * A,"epsilon": epsilon,},index=log_pair.index)

    return {"ticker1": ticker1, "ticker2": ticker2, "alpha": alpha, "beta": beta, "spread": epsilon, "result": result}





