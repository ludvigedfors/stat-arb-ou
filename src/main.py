import LoadData
import matplotlib.pyplot as plt


def main():
    print("Hello from VS Code and GitHub!")

def test():
    print("This is a test function.")


if __name__ == "__main__":
    # main()
    # test()
    pair = LoadData.load_data()
    cointegration_result = LoadData.cointegration_test(pair)
    print(f"Intercept = {cointegration_result['Intercept']:.6f}")
    print(f"Hedge Ratio  = {cointegration_result['HedgeRatio']:.6f}")
    #print(cointegration_result["Result"].head())
    # pair plot for visualization
    # plt.figure(figsize=(12, 6))
    # plt.plot(pair.index, pair.iloc[:, 0], label=pair.columns[0])
    # plt.plot(pair.index, pair.iloc[:, 1], label=pair.columns[1])
    # plt.title(f'Stock Prices of {pair.columns[0]} and {pair.columns[1]}'.format(pair.columns[0], pair.columns[1]))
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()
    # spread plot for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(cointegration_result["Result"].index, cointegration_result["Spread"], label='Spread')
    # draw dotted line at 0 to see mean reverting behavior
    # plt.hline(y=0, color='r', linestyle='--', label='No spread')
    plt.title(f'Spread between {cointegration_result["Ticker A"]} and {cointegration_result["Ticker B"]}')
    plt.xlabel('Date')  
    plt.ylabel('Spread')
    plt.legend()
    plt.show()
